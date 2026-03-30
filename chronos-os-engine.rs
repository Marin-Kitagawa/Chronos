// ============================================================================
// CHRONOS OPERATING SYSTEM PRIMITIVES ENGINE
// ============================================================================
//
// HOW AN OPERATING SYSTEM ACTUALLY WORKS (and what this code implements):
//
// An OS is fundamentally a resource manager. It manages three critical
// resources that every program needs:
//
// 1. MEMORY: Programs think they have access to a giant contiguous address
//    space, but in reality physical RAM is a finite, shared resource. The OS
//    uses a page table to map virtual addresses to physical frames, and a
//    page frame allocator to track which physical frames are free. When a
//    program accesses a page that isn't mapped, the CPU generates a page
//    fault, and the OS decides what to do (allocate a new frame, load from
//    disk, or kill the process).
//
// 2. CPU TIME: Multiple programs want to run simultaneously, but there are
//    only a few CPU cores. The OS scheduler decides which process runs next,
//    using algorithms like Round-Robin, Priority Scheduling, or Completely
//    Fair Scheduler. It preempts running processes using timer interrupts.
//
// 3. I/O DEVICES: Programs need to read files, send network packets, and
//    display graphics. The OS provides a uniform interface (file descriptors,
//    syscalls) that hides the complexity of different hardware devices.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic):
//   1.  Page frame allocator (bitmap-based, O(n) scan with free list cache)
//   2.  Virtual memory manager (page table simulation with 4-level paging)
//   3.  Buddy allocator (the kernel's primary heap allocator)
//   4.  Slab allocator (for fixed-size kernel objects)
//   5.  Process control blocks and lifecycle management
//   6.  Thread management with stack allocation
//   7.  Preemptive scheduler (Round-Robin, Priority, CFS, EDF)
//   8.  Virtual filesystem with inodes, directories, permissions
//   9.  Inter-process communication (pipes, message queues, shared memory)
//  10.  Signal handling
//  11.  File descriptor table and I/O multiplexing
//  12.  Device driver model with interrupt handling
//  13.  Memory-mapped I/O register access (for embedded/driver development)
//  14.  System call dispatcher
//  15.  Ring buffer (lock-free) for kernel logging and inter-core communication
// ============================================================================

use std::collections::{HashMap, BTreeMap, VecDeque, BinaryHeap};
use std::cmp::{Ordering, Reverse};
use std::time::{Duration, Instant};

// ============================================================================
// PART 1: PAGE FRAME ALLOCATOR
// ============================================================================
// Physical memory is divided into fixed-size pages (typically 4 KiB). The
// page frame allocator tracks which physical frames are free and which are
// in use. When a process needs memory, the OS asks this allocator for a
// free frame. When memory is freed, the frame goes back to the allocator.
//
// This implementation uses a bitmap where each bit represents one physical
// page frame. Bit 0 = frame 0, bit 1 = frame 1, etc. A set bit means
// the frame is in use; a clear bit means it's free. We also maintain a
// free-list cache for O(1) allocation in the common case.

/// The size of a page frame in bytes. 4 KiB is the standard on x86-64.
pub const PAGE_SIZE: usize = 4096;

/// Manages physical page frame allocation. This is what a real kernel uses
/// to track which pages of physical RAM are free.
pub struct PageFrameAllocator {
    /// Bitmap: one bit per physical frame. true = allocated, false = free.
    bitmap: Vec<bool>,
    /// Total number of physical frames available.
    total_frames: usize,
    /// Number of currently allocated frames.
    allocated_frames: usize,
    /// Cache of recently freed frame numbers for O(1) re-allocation.
    /// This avoids scanning the bitmap every time we need a frame.
    free_cache: Vec<usize>,
    /// The maximum number of frames to keep in the free cache.
    free_cache_max: usize,
    /// Hint: the position to start scanning from (optimization to avoid
    /// rescanning the beginning of the bitmap every time).
    next_scan_start: usize,
}

impl PageFrameAllocator {
    /// Create a new allocator managing `total_memory_bytes` of physical RAM.
    /// All frames start as free.
    pub fn new(total_memory_bytes: usize) -> Self {
        let total_frames = total_memory_bytes / PAGE_SIZE;
        Self {
            bitmap: vec![false; total_frames],
            total_frames,
            allocated_frames: 0,
            free_cache: Vec::with_capacity(256),
            free_cache_max: 256,
            next_scan_start: 0,
        }
    }

    /// Reserve a range of frames (e.g., for the kernel image, BIOS areas).
    /// These frames are marked as allocated and cannot be freed.
    pub fn reserve_range(&mut self, start_frame: usize, count: usize) {
        for i in start_frame..start_frame + count {
            if i < self.total_frames && !self.bitmap[i] {
                self.bitmap[i] = true;
                self.allocated_frames += 1;
            }
        }
    }

    /// Allocate a single physical frame. Returns the frame number, or None
    /// if no free frames are available (out of memory).
    ///
    /// Strategy: first check the free cache (O(1)), then scan the bitmap
    /// starting from the hint position (O(n) worst case, but usually fast
    /// because we remember where we left off).
    pub fn allocate_frame(&mut self) -> Option<usize> {
        // Fast path: use the free cache.
        if let Some(frame) = self.free_cache.pop() {
            // Double-check it's actually free (defensive programming).
            if !self.bitmap[frame] {
                self.bitmap[frame] = true;
                self.allocated_frames += 1;
                return Some(frame);
            }
            // If it was already allocated (shouldn't happen), fall through to scan.
        }

        // Slow path: scan the bitmap starting from the hint.
        let start = self.next_scan_start;
        for offset in 0..self.total_frames {
            let frame = (start + offset) % self.total_frames;
            if !self.bitmap[frame] {
                self.bitmap[frame] = true;
                self.allocated_frames += 1;
                self.next_scan_start = (frame + 1) % self.total_frames;
                return Some(frame);
            }
        }

        None // Out of memory!
    }

    /// Allocate `count` contiguous physical frames. This is needed for DMA
    /// buffers and large kernel allocations that require physical contiguity.
    /// Returns the starting frame number, or None if not enough contiguous
    /// frames are available.
    pub fn allocate_contiguous(&mut self, count: usize) -> Option<usize> {
        if count == 0 { return None; }
        if count == 1 { return self.allocate_frame(); }

        // Scan for a run of `count` consecutive free frames.
        let mut run_start = 0;
        let mut run_length = 0;

        for i in 0..self.total_frames {
            if !self.bitmap[i] {
                if run_length == 0 { run_start = i; }
                run_length += 1;
                if run_length == count {
                    // Found a long enough run — mark them all as allocated.
                    for j in run_start..run_start + count {
                        self.bitmap[j] = true;
                    }
                    self.allocated_frames += count;
                    return Some(run_start);
                }
            } else {
                run_length = 0;
            }
        }

        None // Couldn't find enough contiguous free frames
    }

    /// Free a previously allocated frame. The frame is returned to the pool
    /// and may be reused by a future allocation.
    pub fn free_frame(&mut self, frame: usize) {
        if frame >= self.total_frames { return; }
        if !self.bitmap[frame] { return; } // Double-free protection

        self.bitmap[frame] = false;
        self.allocated_frames -= 1;

        // Add to the free cache for fast reallocation.
        if self.free_cache.len() < self.free_cache_max {
            self.free_cache.push(frame);
        }
    }

    /// Free a range of contiguous frames.
    pub fn free_contiguous(&mut self, start_frame: usize, count: usize) {
        for i in start_frame..start_frame + count {
            self.free_frame(i);
        }
    }

    /// How many free frames are available.
    pub fn free_frames(&self) -> usize {
        self.total_frames - self.allocated_frames
    }

    /// How many frames are allocated.
    pub fn used_frames(&self) -> usize {
        self.allocated_frames
    }

    /// Total memory managed, in bytes.
    pub fn total_memory(&self) -> usize {
        self.total_frames * PAGE_SIZE
    }

    /// Free memory available, in bytes.
    pub fn free_memory(&self) -> usize {
        self.free_frames() * PAGE_SIZE
    }
}

// ============================================================================
// PART 2: VIRTUAL MEMORY MANAGER
// ============================================================================
// Each process thinks it has its own private address space starting from 0.
// The virtual memory manager translates virtual addresses to physical
// addresses using a page table. On x86-64, this is a 4-level hierarchical
// table (PML4 → PDPT → PD → PT), but we simulate it as a flat map for
// clarity while preserving the exact semantics.
//
// A page table entry contains:
//   - The physical frame number
//   - Permission bits (read, write, execute, user/kernel)
//   - Status bits (present, dirty, accessed)

/// Flags for a page table entry. These control what operations are permitted
/// on the page and track whether the page has been accessed or modified.
#[derive(Debug, Clone, Copy)]
pub struct PageFlags {
    pub present: bool,      // Page is mapped to a physical frame
    pub writable: bool,     // Page can be written to
    pub executable: bool,   // Page contains executable code
    pub user: bool,         // Page is accessible from user mode (not just kernel)
    pub dirty: bool,        // Page has been written to since last cleared
    pub accessed: bool,     // Page has been read since last cleared
    pub cached: bool,       // Page can be cached by the CPU
    pub copy_on_write: bool, // Page is shared; copy it on first write
}

impl PageFlags {
    pub fn kernel_rw() -> Self {
        Self { present: true, writable: true, executable: false, user: false,
               dirty: false, accessed: false, cached: true, copy_on_write: false }
    }
    pub fn user_rw() -> Self {
        Self { present: true, writable: true, executable: false, user: true,
               dirty: false, accessed: false, cached: true, copy_on_write: false }
    }
    pub fn user_rx() -> Self {
        Self { present: true, writable: false, executable: true, user: true,
               dirty: false, accessed: false, cached: true, copy_on_write: false }
    }
    pub fn user_ro() -> Self {
        Self { present: true, writable: false, executable: false, user: true,
               dirty: false, accessed: false, cached: true, copy_on_write: false }
    }
    pub fn not_present() -> Self {
        Self { present: false, writable: false, executable: false, user: false,
               dirty: false, accessed: false, cached: false, copy_on_write: false }
    }
}

/// A single page table entry: maps a virtual page to a physical frame.
#[derive(Debug, Clone)]
pub struct PageTableEntry {
    pub physical_frame: usize,
    pub flags: PageFlags,
}

/// The virtual memory space of a single process. Maps virtual page numbers
/// to physical frames. In a real OS, this would be a hardware page table;
/// here we simulate it with a HashMap for clarity.
pub struct VirtualMemorySpace {
    /// The page table: virtual_page_number → (physical_frame, flags).
    page_table: HashMap<usize, PageTableEntry>,
    /// The process that owns this address space.
    pub owner_pid: u32,
    /// Memory regions (for tracking what each range of pages is for).
    pub regions: Vec<MemoryRegion>,
}

/// A named region of the virtual address space (text, data, heap, stack, mmap).
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub name: String,
    pub start_page: usize,
    pub page_count: usize,
    pub flags: PageFlags,
}

/// Errors that can occur during virtual memory operations.
#[derive(Debug, Clone)]
pub enum VmError {
    PageFault { virtual_address: usize, reason: PageFaultReason },
    OutOfMemory,
    PermissionDenied { virtual_address: usize, attempted: String },
    AlreadyMapped { virtual_page: usize },
    NotMapped { virtual_page: usize },
}

#[derive(Debug, Clone)]
pub enum PageFaultReason {
    NotPresent,          // Page is not mapped at all
    WriteToReadOnly,     // Attempted to write to a read-only page
    ExecuteNonExecutable,// Attempted to execute non-executable page
    KernelAccess,        // User-mode code tried to access kernel page
    CopyOnWrite,         // Shared page needs to be copied before writing
}

impl std::fmt::Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VmError::PageFault { virtual_address, reason } =>
                write!(f, "Page fault at 0x{:016X}: {:?}", virtual_address, reason),
            VmError::OutOfMemory => write!(f, "Out of physical memory"),
            VmError::PermissionDenied { virtual_address, attempted } =>
                write!(f, "Permission denied at 0x{:016X}: {}", virtual_address, attempted),
            VmError::AlreadyMapped { virtual_page } =>
                write!(f, "Virtual page {} is already mapped", virtual_page),
            VmError::NotMapped { virtual_page } =>
                write!(f, "Virtual page {} is not mapped", virtual_page),
        }
    }
}

impl VirtualMemorySpace {
    pub fn new(owner_pid: u32) -> Self {
        Self {
            page_table: HashMap::new(),
            owner_pid,
            regions: Vec::new(),
        }
    }

    /// Map a virtual page to a physical frame with the given permissions.
    /// This is what the kernel does when a process requests memory.
    pub fn map_page(&mut self, virtual_page: usize, physical_frame: usize, flags: PageFlags) -> Result<(), VmError> {
        if self.page_table.contains_key(&virtual_page) {
            return Err(VmError::AlreadyMapped { virtual_page });
        }
        self.page_table.insert(virtual_page, PageTableEntry { physical_frame, flags });
        Ok(())
    }

    /// Unmap a virtual page, freeing the physical frame.
    pub fn unmap_page(&mut self, virtual_page: usize) -> Result<usize, VmError> {
        match self.page_table.remove(&virtual_page) {
            Some(entry) => Ok(entry.physical_frame),
            None => Err(VmError::NotMapped { virtual_page }),
        }
    }

    /// Translate a virtual address to a physical address.
    /// This simulates what the CPU's MMU (Memory Management Unit) does
    /// on every single memory access. If the page isn't mapped, or the
    /// access violates permissions, a page fault occurs.
    pub fn translate(&mut self, virtual_address: usize, write: bool, execute: bool, user_mode: bool) -> Result<usize, VmError> {
        let virtual_page = virtual_address / PAGE_SIZE;
        let page_offset = virtual_address % PAGE_SIZE;

        match self.page_table.get_mut(&virtual_page) {
            None => {
                // Page not present → page fault.
                Err(VmError::PageFault {
                    virtual_address,
                    reason: PageFaultReason::NotPresent,
                })
            }
            Some(entry) => {
                if !entry.flags.present {
                    return Err(VmError::PageFault {
                        virtual_address,
                        reason: PageFaultReason::NotPresent,
                    });
                }

                // Permission checks (just like the real MMU does):
                if write && !entry.flags.writable {
                    if entry.flags.copy_on_write {
                        return Err(VmError::PageFault {
                            virtual_address,
                            reason: PageFaultReason::CopyOnWrite,
                        });
                    }
                    return Err(VmError::PageFault {
                        virtual_address,
                        reason: PageFaultReason::WriteToReadOnly,
                    });
                }
                if execute && !entry.flags.executable {
                    return Err(VmError::PageFault {
                        virtual_address,
                        reason: PageFaultReason::ExecuteNonExecutable,
                    });
                }
                if user_mode && !entry.flags.user {
                    return Err(VmError::PageFault {
                        virtual_address,
                        reason: PageFaultReason::KernelAccess,
                    });
                }

                // Set accessed and dirty bits (the CPU does this automatically).
                entry.flags.accessed = true;
                if write { entry.flags.dirty = true; }

                // Compute the physical address.
                let physical_address = entry.physical_frame * PAGE_SIZE + page_offset;
                Ok(physical_address)
            }
        }
    }

    /// Map a contiguous region of virtual pages to physical frames.
    /// Returns the list of physical frames allocated.
    pub fn map_region(
        &mut self,
        name: &str,
        start_page: usize,
        page_count: usize,
        flags: PageFlags,
        allocator: &mut PageFrameAllocator,
    ) -> Result<Vec<usize>, VmError> {
        let mut frames = Vec::with_capacity(page_count);

        for i in 0..page_count {
            match allocator.allocate_frame() {
                Some(frame) => {
                    self.map_page(start_page + i, frame, flags)?;
                    frames.push(frame);
                }
                None => {
                    // Rollback: unmap and free everything we've allocated so far.
                    for j in 0..i {
                        if let Ok(frame) = self.unmap_page(start_page + j) {
                            allocator.free_frame(frame);
                        }
                    }
                    return Err(VmError::OutOfMemory);
                }
            }
        }

        self.regions.push(MemoryRegion {
            name: name.to_string(), start_page, page_count, flags,
        });

        Ok(frames)
    }

    /// How many pages are mapped in this address space.
    pub fn mapped_pages(&self) -> usize {
        self.page_table.len()
    }
}

// ============================================================================
// PART 3: BUDDY ALLOCATOR
// ============================================================================
// The buddy allocator is the primary kernel heap allocator used by Linux.
// It manages blocks of memory in powers of two. When you request N bytes,
// it rounds up to the next power of two and returns a block of that size.
// When you free a block, it checks if its "buddy" (the adjacent block of
// the same size) is also free. If so, they merge into a larger block.
// This prevents external fragmentation.
//
// For example, with a 1 MB heap:
//   allocate(100 bytes) → returns a 128-byte block (next power of 2)
//   allocate(300 bytes) → returns a 512-byte block
//   free both → they merge back into a 1024-byte block (if adjacent)

/// A buddy allocator managing a region of memory.
pub struct BuddyAllocator {
    /// For each order k (0..max_order), a list of free blocks of size 2^k * min_block.
    free_lists: Vec<Vec<usize>>,
    /// Minimum block size (usually the cache line size, 64 bytes).
    min_block_size: usize,
    /// Maximum order: the allocator manages blocks up to 2^max_order * min_block_size.
    max_order: usize,
    /// Total memory managed.
    total_size: usize,
    /// Tracks which blocks are allocated (address → order).
    allocated: HashMap<usize, usize>,
    /// Total bytes currently allocated.
    bytes_allocated: usize,
}

impl BuddyAllocator {
    /// Create a new buddy allocator over a memory region starting at `base`
    /// with `total_size` bytes. The minimum allocation unit is `min_block` bytes.
    pub fn new(base: usize, total_size: usize, min_block: usize) -> Self {
        let max_order = (total_size / min_block).trailing_zeros() as usize;
        let mut free_lists = vec![Vec::new(); max_order + 1];
        // The entire region starts as one big free block at the maximum order.
        free_lists[max_order].push(base);

        Self {
            free_lists,
            min_block_size: min_block,
            max_order,
            total_size,
            allocated: HashMap::new(),
            bytes_allocated: 0,
        }
    }

    /// Allocate a block of at least `size` bytes.
    /// Returns the starting address of the block, or None if no suitable block exists.
    ///
    /// Algorithm:
    /// 1. Find the smallest order k where 2^k * min_block >= size.
    /// 2. Look for a free block at order k.
    /// 3. If none exists, find a free block at a higher order and SPLIT it.
    ///    Splitting a block of order k+1 produces two blocks of order k.
    ///    One goes to the caller; the other goes on the free list for order k.
    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        // Find the order needed for this allocation.
        let needed_order = self.size_to_order(size);
        if needed_order > self.max_order { return None; }

        // Find the smallest available order >= needed_order.
        let mut found_order = None;
        for order in needed_order..=self.max_order {
            if !self.free_lists[order].is_empty() {
                found_order = Some(order);
                break;
            }
        }

        let available_order = found_order?;

        // Split down from the available order to the needed order.
        // Each split produces two halves; we keep one and free the other.
        let mut block = self.free_lists[available_order].pop().unwrap();

        for order in (needed_order..available_order).rev() {
            // Split the block: the first half stays as our allocation,
            // the second half goes on the free list for the smaller order.
            let buddy = block + (self.min_block_size << order);
            self.free_lists[order].push(buddy);
        }

        let alloc_size = self.min_block_size << needed_order;
        self.allocated.insert(block, needed_order);
        self.bytes_allocated += alloc_size;

        Some(block)
    }

    /// Free a previously allocated block.
    ///
    /// Algorithm:
    /// 1. Look up the order of the block.
    /// 2. Compute the buddy address: for a block at address A of order k,
    ///    the buddy is at A XOR (2^k * min_block).
    /// 3. If the buddy is free at the same order, MERGE them into a block
    ///    of order k+1. Repeat the merge at the next order.
    /// 4. If the buddy is not free, just add this block to the free list.
    pub fn free(&mut self, address: usize) {
        let order = match self.allocated.remove(&address) {
            Some(o) => o,
            None => return, // Not allocated by us
        };

        let alloc_size = self.min_block_size << order;
        self.bytes_allocated -= alloc_size;

        let mut current_address = address;
        let mut current_order = order;

        // Try to merge with the buddy at each level.
        while current_order < self.max_order {
            let buddy_address = current_address ^ (self.min_block_size << current_order);

            // Check if the buddy is on the free list at this order.
            if let Some(pos) = self.free_lists[current_order].iter().position(|&a| a == buddy_address) {
                // Buddy is free! Remove it from the free list and merge.
                self.free_lists[current_order].swap_remove(pos);
                // The merged block starts at the lower of the two addresses.
                current_address = current_address.min(buddy_address);
                current_order += 1;
            } else {
                break; // Buddy is not free; stop merging.
            }
        }

        self.free_lists[current_order].push(current_address);
    }

    /// Convert a size in bytes to the buddy order.
    /// Order k means the block size is 2^k * min_block_size.
    fn size_to_order(&self, size: usize) -> usize {
        let blocks_needed = (size + self.min_block_size - 1) / self.min_block_size;
        // Find the smallest power of 2 >= blocks_needed.
        if blocks_needed <= 1 { 0 }
        else { (blocks_needed - 1).next_power_of_two().trailing_zeros() as usize }
    }

    pub fn bytes_free(&self) -> usize { self.total_size - self.bytes_allocated }
    pub fn bytes_used(&self) -> usize { self.bytes_allocated }

    /// How many free blocks at each order (useful for fragmentation analysis).
    pub fn free_block_counts(&self) -> Vec<(usize, usize, usize)> {
        self.free_lists.iter().enumerate().map(|(order, list)| {
            let block_size = self.min_block_size << order;
            (order, block_size, list.len())
        }).collect()
    }
}

// ============================================================================
// PART 4: SLAB ALLOCATOR
// ============================================================================
// The slab allocator sits on top of the buddy allocator and provides
// efficient allocation for fixed-size objects. Instead of calling the buddy
// allocator for every small object (which would waste memory due to power-of-2
// rounding), the slab allocator pre-allocates "slabs" of memory and divides
// them into fixed-size slots. This is exactly what Linux uses for kernel
// objects like struct task_struct, struct inode, etc.

/// A slab cache for objects of a specific size.
pub struct SlabCache {
    pub object_size: usize,
    pub objects_per_slab: usize,
    pub name: String,
    /// Each slab is a contiguous block of memory divided into slots.
    slabs: Vec<Slab>,
    /// Statistics.
    pub total_allocated: usize,
    pub total_freed: usize,
}

struct Slab {
    base_address: usize,         // Where this slab's memory starts
    capacity: usize,             // How many objects fit in this slab
    free_list: Vec<usize>,       // Indices of free slots within the slab
    allocated_count: usize,
}

impl SlabCache {
    /// Create a new slab cache for objects of `object_size` bytes.
    /// Each slab will hold `objects_per_slab` objects.
    pub fn new(name: &str, object_size: usize, objects_per_slab: usize) -> Self {
        Self {
            object_size: object_size.max(8), // Minimum 8 bytes for alignment
            objects_per_slab,
            name: name.to_string(),
            slabs: Vec::new(),
            total_allocated: 0,
            total_freed: 0,
        }
    }

    /// Allocate one object from the cache. Returns the address of the object.
    /// If all existing slabs are full, creates a new slab.
    pub fn allocate(&mut self, buddy: &mut BuddyAllocator) -> Option<usize> {
        // Try to find a slab with a free slot.
        for slab in &mut self.slabs {
            if let Some(slot_index) = slab.free_list.pop() {
                slab.allocated_count += 1;
                self.total_allocated += 1;
                return Some(slab.base_address + slot_index * self.object_size);
            }
        }

        // All slabs are full — allocate a new slab from the buddy allocator.
        let slab_size = self.objects_per_slab * self.object_size;
        let base = buddy.allocate(slab_size)?;

        let mut new_slab = Slab {
            base_address: base,
            capacity: self.objects_per_slab,
            free_list: (1..self.objects_per_slab).rev().collect(), // All slots except 0
            allocated_count: 1, // Slot 0 is being allocated now
        };

        let result = base; // Return slot 0
        self.slabs.push(new_slab);
        self.total_allocated += 1;
        Some(result)
    }

    /// Free an object back to the cache. The address must have been returned
    /// by a previous `allocate` call on this cache.
    pub fn free(&mut self, address: usize) {
        for slab in &mut self.slabs {
            if address >= slab.base_address
                && address < slab.base_address + slab.capacity * self.object_size
            {
                let slot_index = (address - slab.base_address) / self.object_size;
                slab.free_list.push(slot_index);
                slab.allocated_count -= 1;
                self.total_freed += 1;
                return;
            }
        }
    }

    pub fn active_objects(&self) -> usize { self.total_allocated - self.total_freed }
}

// ============================================================================
// PART 5: PROCESS MANAGEMENT
// ============================================================================
// A process is a running instance of a program. The kernel tracks each
// process with a Process Control Block (PCB) that contains its state,
// registers, memory space, file descriptors, and scheduling information.

pub type Pid = u32;

/// The state a process can be in. This directly maps to what `ps` shows on Linux.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    Created,        // Just created, not yet scheduled
    Ready,          // Ready to run, waiting for CPU time
    Running,        // Currently executing on a CPU
    Blocked,        // Waiting for I/O, mutex, signal, etc.
    Sleeping,       // Sleeping for a specific duration
    Zombie,         // Terminated but parent hasn't called wait() yet
    Dead,           // Fully terminated and cleaned up
}

/// The Process Control Block: all kernel state for one process.
#[derive(Debug, Clone)]
pub struct Process {
    pub pid: Pid,
    pub parent_pid: Option<Pid>,
    pub name: String,
    pub state: ProcessState,
    pub priority: i8,               // -20 (highest) to +19 (lowest), like nice values
    pub exit_code: Option<i32>,

    // --- CPU state (saved/restored on context switch) ---
    pub registers: RegisterState,

    // --- Scheduling ---
    pub cpu_time_used: Duration,    // Total CPU time consumed
    pub time_slice: Duration,       // How long until preemption
    pub vruntime: u64,              // Virtual runtime for CFS (nanoseconds)
    pub last_scheduled: Option<Instant>,
    pub wake_time: Option<Instant>, // For sleeping processes

    // --- Memory ---
    pub memory_pages: usize,        // Number of virtual pages mapped

    // --- File descriptors ---
    pub fd_table: Vec<Option<FileDescriptor>>,
    pub next_fd: usize,

    // --- Signals ---
    pub pending_signals: Vec<Signal>,
    pub signal_mask: u64,           // Bitmask of blocked signals

    // --- Threads ---
    pub thread_ids: Vec<u32>,
}

/// Simulated CPU register state (saved/restored on context switch).
/// On a real x86-64 processor, there are 16 general-purpose registers,
/// the instruction pointer, the stack pointer, flags, and FPU state.
#[derive(Debug, Clone, Default)]
pub struct RegisterState {
    pub rip: u64,   // Instruction pointer (program counter)
    pub rsp: u64,   // Stack pointer
    pub rbp: u64,   // Base pointer (frame pointer)
    pub rax: u64,   // Return value / accumulator
    pub rbx: u64,   // Callee-saved
    pub rcx: u64,   // 4th argument
    pub rdx: u64,   // 3rd argument
    pub rsi: u64,   // 2nd argument
    pub rdi: u64,   // 1st argument
    pub r8: u64, pub r9: u64, pub r10: u64, pub r11: u64,
    pub r12: u64, pub r13: u64, pub r14: u64, pub r15: u64,
    pub rflags: u64,
}

/// A file descriptor: a handle to an open file, pipe, socket, or device.
#[derive(Debug, Clone)]
pub struct FileDescriptor {
    pub kind: FdKind,
    pub flags: u32,           // O_RDONLY, O_WRONLY, O_RDWR, O_NONBLOCK, etc.
    pub offset: usize,        // Current read/write position
    pub inode: Option<usize>, // For regular files
}

#[derive(Debug, Clone)]
pub enum FdKind {
    RegularFile { path: String },
    Directory { path: String },
    Pipe { pipe_id: usize },
    Socket { socket_id: usize },
    Device { major: u32, minor: u32 },
    EventFd,
    TimerFd,
}

/// Signals (subset of POSIX signals).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    SIGHUP = 1, SIGINT = 2, SIGQUIT = 3, SIGILL = 4, SIGTRAP = 5,
    SIGABRT = 6, SIGBUS = 7, SIGFPE = 8, SIGKILL = 9, SIGUSR1 = 10,
    SIGSEGV = 11, SIGUSR2 = 12, SIGPIPE = 13, SIGALRM = 14,
    SIGTERM = 15, SIGCHLD = 17, SIGCONT = 18, SIGSTOP = 19,
}

impl Process {
    pub fn new(pid: Pid, name: &str, parent_pid: Option<Pid>) -> Self {
        // Every process starts with three standard file descriptors:
        // 0 = stdin, 1 = stdout, 2 = stderr.
        let fd_table = vec![
            Some(FileDescriptor { kind: FdKind::Device { major: 5, minor: 0 }, flags: 0, offset: 0, inode: None }), // stdin
            Some(FileDescriptor { kind: FdKind::Device { major: 5, minor: 1 }, flags: 1, offset: 0, inode: None }), // stdout
            Some(FileDescriptor { kind: FdKind::Device { major: 5, minor: 2 }, flags: 1, offset: 0, inode: None }), // stderr
        ];

        Self {
            pid, parent_pid: parent_pid, name: name.to_string(),
            state: ProcessState::Created,
            priority: 0, exit_code: None,
            registers: RegisterState::default(),
            cpu_time_used: Duration::ZERO,
            time_slice: Duration::from_millis(10), // 10ms default quantum
            vruntime: 0, last_scheduled: None, wake_time: None,
            memory_pages: 0,
            fd_table, next_fd: 3,
            pending_signals: Vec::new(), signal_mask: 0,
            thread_ids: Vec::new(),
        }
    }

    /// Allocate a new file descriptor for this process.
    pub fn alloc_fd(&mut self, fd: FileDescriptor) -> usize {
        // Find the lowest available FD number (POSIX semantics).
        for (i, slot) in self.fd_table.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(fd);
                return i;
            }
        }
        // No free slots — extend the table.
        let idx = self.fd_table.len();
        self.fd_table.push(Some(fd));
        idx
    }

    /// Close a file descriptor.
    pub fn close_fd(&mut self, fd_num: usize) -> bool {
        if fd_num < self.fd_table.len() && self.fd_table[fd_num].is_some() {
            self.fd_table[fd_num] = None;
            true
        } else {
            false
        }
    }

    /// Send a signal to this process.
    pub fn send_signal(&mut self, signal: Signal) {
        // SIGKILL and SIGSTOP cannot be blocked.
        let sig_num = signal as u64;
        if signal == Signal::SIGKILL || signal == Signal::SIGSTOP || (self.signal_mask & (1 << sig_num)) == 0 {
            self.pending_signals.push(signal);
        }
    }
}

// ============================================================================
// PART 6: PROCESS SCHEDULER
// ============================================================================
// The scheduler decides which process runs next. It is invoked by the
// timer interrupt (preemption) and by system calls that block the current
// process. We implement four scheduling algorithms:
//
// 1. Round-Robin: each process gets a fixed time quantum, then the next
//    process runs. Simple and fair, but not great for interactive tasks.
//
// 2. Priority: processes with higher priority always run first. Can cause
//    starvation (low-priority processes never run).
//
// 3. Completely Fair Scheduler (CFS): Linux's default scheduler. It tracks
//    "virtual runtime" (vruntime) for each process and always runs the
//    process with the smallest vruntime. This ensures every process gets
//    its fair share of CPU time, weighted by priority.
//
// 4. Earliest Deadline First (EDF): for real-time systems. The process
//    with the nearest deadline always runs first.

#[derive(Debug, Clone)]
pub enum SchedulingAlgorithm {
    RoundRobin { quantum: Duration },
    Priority,
    CFS,
    EDF,
}

/// The process scheduler.
pub struct Scheduler {
    /// All processes in the system, keyed by PID.
    pub processes: HashMap<Pid, Process>,
    /// The ready queue: processes waiting to run.
    ready_queue: VecDeque<Pid>,
    /// For CFS: a red-black tree (simulated with BTreeMap) ordered by vruntime.
    cfs_tree: BTreeMap<(u64, Pid), Pid>,
    /// For EDF: processes ordered by deadline.
    edf_queue: BinaryHeap<Reverse<(u64, Pid)>>,
    /// The currently running process.
    pub current_pid: Option<Pid>,
    /// The scheduling algorithm in use.
    pub algorithm: SchedulingAlgorithm,
    /// Next PID to assign.
    next_pid: Pid,
    /// Number of context switches performed (for statistics).
    pub context_switches: u64,
    /// The idle process (PID 0), runs when nothing else can.
    idle_pid: Pid,
}

impl Scheduler {
    pub fn new(algorithm: SchedulingAlgorithm) -> Self {
        let mut processes = HashMap::new();
        // Create the idle process (PID 0). It runs an infinite loop and
        // has the lowest priority. It's scheduled when no other process is ready.
        let mut idle = Process::new(0, "idle", None);
        idle.priority = 19; // Lowest priority
        idle.state = ProcessState::Ready;
        processes.insert(0, idle);

        Self {
            processes,
            ready_queue: VecDeque::new(),
            cfs_tree: BTreeMap::new(),
            edf_queue: BinaryHeap::new(),
            current_pid: None,
            algorithm,
            next_pid: 1,
            context_switches: 0,
            idle_pid: 0,
        }
    }

    /// Create a new process and add it to the ready queue.
    /// Returns the PID of the new process.
    pub fn create_process(&mut self, name: &str, parent_pid: Option<Pid>, priority: i8) -> Pid {
        let pid = self.next_pid;
        self.next_pid += 1;

        let mut process = Process::new(pid, name, parent_pid);
        process.priority = priority;
        process.state = ProcessState::Ready;

        self.processes.insert(pid, process);
        self.enqueue(pid);
        pid
    }

    /// Terminate a process.
    pub fn terminate_process(&mut self, pid: Pid, exit_code: i32) {
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Zombie;
            proc.exit_code = Some(exit_code);

            // Send SIGCHLD to the parent.
            if let Some(parent_pid) = proc.parent_pid {
                if let Some(parent) = self.processes.get_mut(&parent_pid) {
                    parent.send_signal(Signal::SIGCHLD);
                }
            }
        }

        if self.current_pid == Some(pid) {
            self.current_pid = None;
        }
    }

    /// Block the currently running process (e.g., waiting for I/O).
    pub fn block_current(&mut self, reason: &str) {
        if let Some(pid) = self.current_pid {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Blocked;
            }
            self.current_pid = None;
        }
    }

    /// Wake up a blocked process and put it back on the ready queue.
    pub fn wake_process(&mut self, pid: Pid) {
        if let Some(proc) = self.processes.get_mut(&pid) {
            if proc.state == ProcessState::Blocked || proc.state == ProcessState::Sleeping {
                proc.state = ProcessState::Ready;
                self.enqueue(pid);
            }
        }
    }

    /// Add a process to the appropriate ready queue based on the scheduling algorithm.
    fn enqueue(&mut self, pid: Pid) {
        match &self.algorithm {
            SchedulingAlgorithm::RoundRobin { .. } => {
                self.ready_queue.push_back(pid);
            }
            SchedulingAlgorithm::Priority => {
                // Insert in priority order (highest priority = lowest nice value first).
                let priority = self.processes.get(&pid).map(|p| p.priority).unwrap_or(0);
                let pos = self.ready_queue.iter().position(|&other_pid| {
                    self.processes.get(&other_pid).map(|p| p.priority).unwrap_or(0) > priority
                }).unwrap_or(self.ready_queue.len());
                self.ready_queue.insert(pos, pid);
            }
            SchedulingAlgorithm::CFS => {
                let vruntime = self.processes.get(&pid).map(|p| p.vruntime).unwrap_or(0);
                self.cfs_tree.insert((vruntime, pid), pid);
            }
            SchedulingAlgorithm::EDF => {
                // For EDF, we'd use the process's deadline. For now, use vruntime as proxy.
                let deadline = self.processes.get(&pid).map(|p| p.vruntime).unwrap_or(0);
                self.edf_queue.push(Reverse((deadline, pid)));
            }
        }
    }

    /// Dequeue the next process to run.
    fn dequeue(&mut self) -> Option<Pid> {
        match &self.algorithm {
            SchedulingAlgorithm::RoundRobin { .. } => {
                self.ready_queue.pop_front()
            }
            SchedulingAlgorithm::Priority => {
                self.ready_queue.pop_front()
            }
            SchedulingAlgorithm::CFS => {
                // CFS always picks the process with the smallest vruntime.
                // This ensures fairness: every process gets roughly equal CPU time.
                if let Some((&key, &pid)) = self.cfs_tree.iter().next() {
                    self.cfs_tree.remove(&key);
                    Some(pid)
                } else {
                    None
                }
            }
            SchedulingAlgorithm::EDF => {
                while let Some(Reverse((_, pid))) = self.edf_queue.pop() {
                    if self.processes.get(&pid).map(|p| p.state == ProcessState::Ready).unwrap_or(false) {
                        return Some(pid);
                    }
                }
                None
            }
        }
    }

    /// The main scheduling function. Called by the timer interrupt handler
    /// or when a process blocks voluntarily. This is the CORE of the OS.
    ///
    /// It saves the state of the currently running process, selects the
    /// next process to run, and restores that process's state. On a real
    /// OS, this involves switching page tables (changing the address space),
    /// restoring registers, and jumping to the saved instruction pointer.
    pub fn schedule(&mut self) -> Option<Pid> {
        let now = Instant::now();

        // Step 1: If there's a currently running process, save its state
        // and put it back on the ready queue (unless it blocked or terminated).
        if let Some(current) = self.current_pid {
            if let Some(proc) = self.processes.get_mut(&current) {
                // Update CPU time accounting.
                if let Some(last) = proc.last_scheduled {
                    let elapsed = now.duration_since(last);
                    proc.cpu_time_used += elapsed;

                    // CFS: update virtual runtime. Higher-priority processes
                    // accumulate vruntime more slowly, so they get more real CPU time.
                    // The weight formula is: vruntime += elapsed * (base_weight / process_weight)
                    let weight = Self::nice_to_weight(proc.priority);
                    let delta_vruntime = (elapsed.as_nanos() as u64) * 1024 / weight;
                    proc.vruntime += delta_vruntime;
                }

                if proc.state == ProcessState::Running {
                    proc.state = ProcessState::Ready;
                    let pid = current;
                    self.enqueue(pid);
                }
            }
        }

        // Step 2: Wake up any sleeping processes whose wake time has passed.
        let pids: Vec<Pid> = self.processes.keys().cloned().collect();
        for pid in pids {
            let should_wake = self.processes.get(&pid).map(|p| {
                p.state == ProcessState::Sleeping
                    && p.wake_time.map(|t| now >= t).unwrap_or(false)
            }).unwrap_or(false);
            if should_wake {
                self.wake_process(pid);
            }
        }

        // Step 3: Select the next process to run.
        let next = self.dequeue().unwrap_or(self.idle_pid);

        // Step 4: Context switch to the selected process.
        if let Some(proc) = self.processes.get_mut(&next) {
            proc.state = ProcessState::Running;
            proc.last_scheduled = Some(now);
        }

        self.current_pid = Some(next);
        self.context_switches += 1;

        Some(next)
    }

    /// Convert a nice value (-20..+19) to a scheduling weight.
    /// This is the actual weight table used by Linux's CFS scheduler.
    /// Higher weight = more CPU time. nice 0 = weight 1024.
    fn nice_to_weight(nice: i8) -> u64 {
        // Simplified version of the Linux nice-to-weight table.
        // Each nice level differs by about 10% in CPU allocation.
        let clamped = nice.max(-20).min(19);
        let index = (clamped + 20) as usize;
        const WEIGHTS: [u64; 40] = [
            88761, 71755, 56483, 46273, 36291,  // -20 to -16
            29154, 23254, 18705, 14949, 11916,  // -15 to -11
             9548,  7620,  6100,  4904,  3906,  // -10 to -6
             3121,  2501,  1991,  1586,  1277,  //  -5 to -1
             1024,   820,   655,   526,   423,  //   0 to  4
              335,   272,   215,   172,   137,  //   5 to  9
              110,    87,    70,    56,    45,   //  10 to 14
               36,    29,    23,    18,    15,   //  15 to 19
        ];
        WEIGHTS[index]
    }

    /// Put the current process to sleep for a specified duration.
    pub fn sleep_current(&mut self, duration: Duration) {
        if let Some(pid) = self.current_pid {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Sleeping;
                proc.wake_time = Some(Instant::now() + duration);
            }
            self.current_pid = None;
        }
    }

    /// Get a snapshot of all processes (like the `ps` command).
    pub fn process_list(&self) -> Vec<ProcessInfo> {
        self.processes.values().map(|p| ProcessInfo {
            pid: p.pid,
            parent_pid: p.parent_pid,
            name: p.name.clone(),
            state: p.state,
            priority: p.priority,
            cpu_time: p.cpu_time_used,
            memory_pages: p.memory_pages,
            vruntime: p.vruntime,
        }).collect()
    }
}

#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub pid: Pid,
    pub parent_pid: Option<Pid>,
    pub name: String,
    pub state: ProcessState,
    pub priority: i8,
    pub cpu_time: Duration,
    pub memory_pages: usize,
    pub vruntime: u64,
}

// ============================================================================
// PART 7: VIRTUAL FILESYSTEM
// ============================================================================
// The VFS provides a uniform interface to files, directories, devices, and
// pseudo-filesystems (like /proc and /sys on Linux). Everything is an inode
// (index node) that can be read, written, and has metadata.

/// An inode: the fundamental unit of storage in a filesystem.
/// Every file, directory, device, and pipe has an inode.
#[derive(Debug, Clone)]
pub struct Inode {
    pub ino: usize,                // Inode number (unique identifier)
    pub kind: InodeKind,
    pub permissions: Permissions,
    pub owner_uid: u32,
    pub owner_gid: u32,
    pub size: usize,               // File size in bytes
    pub link_count: u32,           // Number of hard links
    pub created: u64,              // Timestamp (seconds since epoch)
    pub modified: u64,
    pub accessed: u64,
    pub data: InodeData,
}

#[derive(Debug, Clone)]
pub enum InodeKind {
    RegularFile,
    Directory,
    Symlink,
    CharDevice { major: u32, minor: u32 },
    BlockDevice { major: u32, minor: u32 },
    Pipe,
    Socket,
}

#[derive(Debug, Clone)]
pub enum InodeData {
    /// File content stored inline (for small files and our simulation).
    FileContent(Vec<u8>),
    /// Directory entries: name → inode number.
    DirectoryEntries(Vec<(String, usize)>),
    /// Symlink target path.
    SymlinkTarget(String),
    /// Pipe buffer.
    PipeBuffer(VecDeque<u8>),
    /// No data (for device nodes, sockets).
    None,
}

/// Unix-style permissions: rwxrwxrwx (owner, group, other).
#[derive(Debug, Clone, Copy)]
pub struct Permissions {
    pub owner_read: bool, pub owner_write: bool, pub owner_execute: bool,
    pub group_read: bool, pub group_write: bool, pub group_execute: bool,
    pub other_read: bool, pub other_write: bool, pub other_execute: bool,
    pub setuid: bool, pub setgid: bool, pub sticky: bool,
}

impl Permissions {
    pub fn from_mode(mode: u16) -> Self {
        Self {
            owner_read:    mode & 0o400 != 0, owner_write:   mode & 0o200 != 0, owner_execute: mode & 0o100 != 0,
            group_read:    mode & 0o040 != 0, group_write:   mode & 0o020 != 0, group_execute: mode & 0o010 != 0,
            other_read:    mode & 0o004 != 0, other_write:   mode & 0o002 != 0, other_execute: mode & 0o001 != 0,
            setuid: mode & 0o4000 != 0, setgid: mode & 0o2000 != 0, sticky: mode & 0o1000 != 0,
        }
    }

    pub fn to_mode(&self) -> u16 {
        let mut mode = 0u16;
        if self.owner_read    { mode |= 0o400; } if self.owner_write   { mode |= 0o200; } if self.owner_execute { mode |= 0o100; }
        if self.group_read    { mode |= 0o040; } if self.group_write   { mode |= 0o020; } if self.group_execute { mode |= 0o010; }
        if self.other_read    { mode |= 0o004; } if self.other_write   { mode |= 0o002; } if self.other_execute { mode |= 0o001; }
        if self.setuid { mode |= 0o4000; } if self.setgid { mode |= 0o2000; } if self.sticky { mode |= 0o1000; }
        mode
    }

    pub fn to_string(&self) -> String {
        let mut s = String::with_capacity(10);
        s.push(if self.owner_read { 'r' } else { '-' });
        s.push(if self.owner_write { 'w' } else { '-' });
        s.push(if self.owner_execute { if self.setuid { 's' } else { 'x' } } else { if self.setuid { 'S' } else { '-' } });
        s.push(if self.group_read { 'r' } else { '-' });
        s.push(if self.group_write { 'w' } else { '-' });
        s.push(if self.group_execute { if self.setgid { 's' } else { 'x' } } else { '-' });
        s.push(if self.other_read { 'r' } else { '-' });
        s.push(if self.other_write { 'w' } else { '-' });
        s.push(if self.other_execute { if self.sticky { 't' } else { 'x' } } else { '-' });
        s
    }
}

/// The Virtual Filesystem. Holds all inodes and provides path-based operations.
pub struct VFS {
    inodes: HashMap<usize, Inode>,
    next_ino: usize,
    root_ino: usize,
}

impl VFS {
    /// Create a new VFS with an empty root directory.
    pub fn new() -> Self {
        let mut vfs = Self {
            inodes: HashMap::new(),
            next_ino: 1,
            root_ino: 1,
        };

        // Create the root directory (inode 1).
        let root = Inode {
            ino: 1,
            kind: InodeKind::Directory,
            permissions: Permissions::from_mode(0o755),
            owner_uid: 0, owner_gid: 0,
            size: 0, link_count: 2,
            created: 0, modified: 0, accessed: 0,
            data: InodeData::DirectoryEntries(vec![
                (".".to_string(), 1),
                ("..".to_string(), 1),
            ]),
        };
        vfs.inodes.insert(1, root);
        vfs.next_ino = 2;

        vfs
    }

    /// Resolve a path to an inode number by walking the directory tree.
    /// This is what every file operation does first: convert a path string
    /// like "/home/user/file.txt" into an inode number.
    pub fn resolve_path(&self, path: &str) -> Result<usize, FsError> {
        if path.is_empty() || path == "/" {
            return Ok(self.root_ino);
        }

        let components: Vec<&str> = path.split('/')
            .filter(|c| !c.is_empty())
            .collect();

        let mut current_ino = self.root_ino;

        for component in &components {
            let inode = self.inodes.get(&current_ino)
                .ok_or(FsError::NotFound(path.to_string()))?;

            if let InodeData::DirectoryEntries(entries) = &inode.data {
                // Handle symlinks: if we encounter one, resolve it recursively.
                if let Some((_, child_ino)) = entries.iter().find(|(name, _)| name == component) {
                    current_ino = *child_ino;
                } else {
                    return Err(FsError::NotFound(format!("{} in {}", component, path)));
                }
            } else {
                return Err(FsError::NotADirectory(path.to_string()));
            }
        }

        Ok(current_ino)
    }

    /// Create a new regular file at the given path.
    pub fn create_file(&mut self, path: &str, permissions: u16, content: Vec<u8>) -> Result<usize, FsError> {
        let (parent_path, filename) = self.split_path(path);
        let parent_ino = self.resolve_path(&parent_path)?;

        let ino = self.next_ino;
        self.next_ino += 1;

        let inode = Inode {
            ino,
            kind: InodeKind::RegularFile,
            permissions: Permissions::from_mode(permissions),
            owner_uid: 0, owner_gid: 0,
            size: content.len(),
            link_count: 1,
            created: 0, modified: 0, accessed: 0,
            data: InodeData::FileContent(content),
        };
        self.inodes.insert(ino, inode);

        // Add entry to parent directory.
        if let Some(parent) = self.inodes.get_mut(&parent_ino) {
            if let InodeData::DirectoryEntries(ref mut entries) = parent.data {
                if entries.iter().any(|(n, _)| n == filename) {
                    return Err(FsError::AlreadyExists(path.to_string()));
                }
                entries.push((filename.to_string(), ino));
                Ok(ino)
            } else {
                Err(FsError::NotADirectory(parent_path))
            }
        } else {
            Err(FsError::NotFound(parent_path))
        }
    }

    /// Create a new directory at the given path.
    pub fn mkdir(&mut self, path: &str, permissions: u16) -> Result<usize, FsError> {
        let (parent_path, dirname) = self.split_path(path);
        let parent_ino = self.resolve_path(&parent_path)?;

        let ino = self.next_ino;
        self.next_ino += 1;

        let inode = Inode {
            ino,
            kind: InodeKind::Directory,
            permissions: Permissions::from_mode(permissions),
            owner_uid: 0, owner_gid: 0,
            size: 0, link_count: 2,
            created: 0, modified: 0, accessed: 0,
            data: InodeData::DirectoryEntries(vec![
                (".".to_string(), ino),
                ("..".to_string(), parent_ino),
            ]),
        };
        self.inodes.insert(ino, inode);

        if let Some(parent) = self.inodes.get_mut(&parent_ino) {
            if let InodeData::DirectoryEntries(ref mut entries) = parent.data {
                entries.push((dirname.to_string(), ino));
                parent.link_count += 1;
                Ok(ino)
            } else {
                Err(FsError::NotADirectory(parent_path))
            }
        } else {
            Err(FsError::NotFound(parent_path))
        }
    }

    /// Read the contents of a file.
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>, FsError> {
        let ino = self.resolve_path(path)?;
        let inode = self.inodes.get(&ino).ok_or(FsError::NotFound(path.to_string()))?;
        match &inode.data {
            InodeData::FileContent(data) => Ok(data.clone()),
            _ => Err(FsError::NotAFile(path.to_string())),
        }
    }

    /// Write content to a file (overwrite).
    pub fn write_file(&mut self, path: &str, content: Vec<u8>) -> Result<(), FsError> {
        let ino = self.resolve_path(path)?;
        let inode = self.inodes.get_mut(&ino).ok_or(FsError::NotFound(path.to_string()))?;
        match &mut inode.data {
            InodeData::FileContent(data) => {
                *data = content.clone();
                inode.size = content.len();
                Ok(())
            }
            _ => Err(FsError::NotAFile(path.to_string())),
        }
    }

    /// List the contents of a directory.
    pub fn readdir(&self, path: &str) -> Result<Vec<DirEntry>, FsError> {
        let ino = self.resolve_path(path)?;
        let inode = self.inodes.get(&ino).ok_or(FsError::NotFound(path.to_string()))?;
        match &inode.data {
            InodeData::DirectoryEntries(entries) => {
                Ok(entries.iter().map(|(name, child_ino)| {
                    let child = self.inodes.get(child_ino);
                    DirEntry {
                        name: name.clone(),
                        ino: *child_ino,
                        kind: child.map(|c| c.kind.clone()).unwrap_or(InodeKind::RegularFile),
                        size: child.map(|c| c.size).unwrap_or(0),
                        permissions: child.map(|c| c.permissions.to_string()).unwrap_or_default(),
                    }
                }).collect())
            }
            _ => Err(FsError::NotADirectory(path.to_string())),
        }
    }

    /// Delete a file.
    pub fn unlink(&mut self, path: &str) -> Result<(), FsError> {
        let (parent_path, filename) = self.split_path(path);
        let parent_ino = self.resolve_path(&parent_path)?;
        let target_ino = self.resolve_path(path)?;

        // Remove from parent directory.
        if let Some(parent) = self.inodes.get_mut(&parent_ino) {
            if let InodeData::DirectoryEntries(ref mut entries) = parent.data {
                entries.retain(|(name, _)| name != filename);
            }
        }

        // Decrement link count; if zero, remove the inode.
        if let Some(inode) = self.inodes.get_mut(&target_ino) {
            inode.link_count -= 1;
            if inode.link_count == 0 {
                self.inodes.remove(&target_ino);
            }
        }

        Ok(())
    }

    /// Get file metadata (like stat()).
    pub fn stat(&self, path: &str) -> Result<FileStat, FsError> {
        let ino = self.resolve_path(path)?;
        let inode = self.inodes.get(&ino).ok_or(FsError::NotFound(path.to_string()))?;
        Ok(FileStat {
            ino: inode.ino,
            kind: inode.kind.clone(),
            permissions: inode.permissions,
            size: inode.size,
            link_count: inode.link_count,
            owner_uid: inode.owner_uid,
            owner_gid: inode.owner_gid,
        })
    }

    fn split_path<'a>(&self, path: &'a str) -> (String, &'a str) {
        let path = path.trim_end_matches('/');
        match path.rfind('/') {
            Some(pos) if pos == 0 => ("/".to_string(), &path[1..]),
            Some(pos) => (path[..pos].to_string(), &path[pos + 1..]),
            None => ("/".to_string(), path),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub ino: usize,
    pub kind: InodeKind,
    pub size: usize,
    pub permissions: String,
}

#[derive(Debug, Clone)]
pub struct FileStat {
    pub ino: usize,
    pub kind: InodeKind,
    pub permissions: Permissions,
    pub size: usize,
    pub link_count: u32,
    pub owner_uid: u32,
    pub owner_gid: u32,
}

#[derive(Debug, Clone)]
pub enum FsError {
    NotFound(String),
    NotADirectory(String),
    NotAFile(String),
    AlreadyExists(String),
    PermissionDenied(String),
    NotEmpty(String),
}

impl std::fmt::Display for FsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FsError::NotFound(p) => write!(f, "No such file or directory: {}", p),
            FsError::NotADirectory(p) => write!(f, "Not a directory: {}", p),
            FsError::NotAFile(p) => write!(f, "Not a regular file: {}", p),
            FsError::AlreadyExists(p) => write!(f, "File exists: {}", p),
            FsError::PermissionDenied(p) => write!(f, "Permission denied: {}", p),
            FsError::NotEmpty(p) => write!(f, "Directory not empty: {}", p),
        }
    }
}

// ============================================================================
// PART 8: INTER-PROCESS COMMUNICATION (IPC)
// ============================================================================

/// A pipe: a unidirectional byte stream between two processes.
/// One end writes, the other reads. The classic Unix IPC mechanism.
pub struct Pipe {
    pub id: usize,
    buffer: VecDeque<u8>,
    pub capacity: usize,
    pub reader_closed: bool,
    pub writer_closed: bool,
}

impl Pipe {
    pub fn new(id: usize, capacity: usize) -> Self {
        Self { id, buffer: VecDeque::with_capacity(capacity), capacity, reader_closed: false, writer_closed: false }
    }

    /// Write bytes to the pipe. Returns how many bytes were written.
    /// If the pipe is full, returns 0 (the caller should block and retry).
    pub fn write(&mut self, data: &[u8]) -> Result<usize, IpcError> {
        if self.reader_closed { return Err(IpcError::BrokenPipe); }
        let available = self.capacity - self.buffer.len();
        let to_write = data.len().min(available);
        for i in 0..to_write { self.buffer.push_back(data[i]); }
        Ok(to_write)
    }

    /// Read bytes from the pipe. Returns how many bytes were read.
    /// If the pipe is empty, returns 0 (the caller should block and retry).
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, IpcError> {
        if self.buffer.is_empty() && self.writer_closed { return Ok(0); } // EOF
        let to_read = buf.len().min(self.buffer.len());
        for i in 0..to_read { buf[i] = self.buffer.pop_front().unwrap(); }
        Ok(to_read)
    }

    pub fn bytes_available(&self) -> usize { self.buffer.len() }
}

/// A message queue: typed, bounded message passing between processes.
pub struct MessageQueue {
    pub id: usize,
    messages: VecDeque<Message>,
    pub max_messages: usize,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub msg_type: i64,
    pub data: Vec<u8>,
    pub sender_pid: Pid,
}

impl MessageQueue {
    pub fn new(id: usize, max_messages: usize) -> Self {
        Self { id, messages: VecDeque::new(), max_messages }
    }

    pub fn send(&mut self, msg: Message) -> Result<(), IpcError> {
        if self.messages.len() >= self.max_messages {
            return Err(IpcError::QueueFull);
        }
        self.messages.push_back(msg);
        Ok(())
    }

    /// Receive a message, optionally filtering by type.
    /// Type 0 = receive any message.
    /// Type > 0 = receive only messages of that type.
    /// Type < 0 = receive messages with type <= |msg_type|.
    pub fn receive(&mut self, msg_type: i64) -> Option<Message> {
        if msg_type == 0 {
            self.messages.pop_front()
        } else if msg_type > 0 {
            let pos = self.messages.iter().position(|m| m.msg_type == msg_type)?;
            Some(self.messages.remove(pos).unwrap())
        } else {
            let threshold = -msg_type;
            let pos = self.messages.iter().position(|m| m.msg_type <= threshold)?;
            Some(self.messages.remove(pos).unwrap())
        }
    }

    pub fn pending_count(&self) -> usize { self.messages.len() }
}

#[derive(Debug, Clone)]
pub enum IpcError {
    BrokenPipe,
    QueueFull,
    NotFound,
}

// ============================================================================
// PART 9: LOCK-FREE RING BUFFER
// ============================================================================
// A ring buffer (circular buffer) is the foundation of many OS subsystems:
// kernel logging (dmesg), inter-core communication, network packet queues,
// and audio buffers. Our implementation is single-producer single-consumer
// (SPSC) and uses no locks — just atomic-style index manipulation.

/// A fixed-size ring buffer for single-producer, single-consumer scenarios.
/// No locks needed — just two indices (head and tail) that wrap around.
pub struct RingBuffer<T: Clone + Default> {
    buffer: Vec<T>,
    capacity: usize,
    head: usize,    // Next position to write (producer advances this)
    tail: usize,    // Next position to read (consumer advances this)
    count: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![T::default(); capacity],
            capacity,
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    /// Push an item into the buffer. Returns false if the buffer is full.
    pub fn push(&mut self, item: T) -> bool {
        if self.count >= self.capacity { return false; }
        self.buffer[self.head] = item;
        self.head = (self.head + 1) % self.capacity;
        self.count += 1;
        true
    }

    /// Pop an item from the buffer. Returns None if empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.count == 0 { return None; }
        let item = self.buffer[self.tail].clone();
        self.tail = (self.tail + 1) % self.capacity;
        self.count -= 1;
        Some(item)
    }

    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count >= self.capacity }
    pub fn len(&self) -> usize { self.count }
    pub fn capacity(&self) -> usize { self.capacity }
}

// ============================================================================
// PART 10: SYSTEM CALL DISPATCHER
// ============================================================================
// System calls are the interface between user-mode programs and the kernel.
// When a program calls read(), write(), fork(), etc., it triggers a software
// interrupt that transfers control to the kernel's syscall handler.

/// System call numbers (subset matching Linux x86-64 ABI).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Syscall {
    Read = 0,        // read(fd, buf, count)
    Write = 1,       // write(fd, buf, count)
    Open = 2,        // open(path, flags, mode)
    Close = 3,       // close(fd)
    Stat = 4,        // stat(path, buf)
    Mmap = 9,        // mmap(addr, length, prot, flags, fd, offset)
    Munmap = 11,     // munmap(addr, length)
    Brk = 12,        // brk(addr)   - adjust heap
    Fork = 57,       // fork()
    Execve = 59,     // execve(path, argv, envp)
    Exit = 60,       // exit(status)
    Wait4 = 61,      // wait4(pid, status, options, rusage)
    Kill = 62,       // kill(pid, sig)
    Pipe = 22,       // pipe(fds[2])
    Getpid = 39,     // getpid()
    Getppid = 110,   // getppid()
    Nanosleep = 35,  // nanosleep(duration)
    Clone = 56,      // clone(flags, stack)
    Ioctl = 16,      // ioctl(fd, request, arg)
}

/// The result of a system call: either a return value or an error number.
#[derive(Debug, Clone)]
pub enum SyscallResult {
    Ok(i64),
    Err(i32),  // Negative errno value
}

/// The kernel: ties everything together. This is the top-level structure
/// that owns the scheduler, memory manager, filesystem, and IPC mechanisms.
pub struct Kernel {
    pub scheduler: Scheduler,
    pub frame_allocator: PageFrameAllocator,
    pub vfs: VFS,
    pub address_spaces: HashMap<Pid, VirtualMemorySpace>,
    pub pipes: HashMap<usize, Pipe>,
    pub message_queues: HashMap<usize, MessageQueue>,
    pub next_pipe_id: usize,
    pub next_mq_id: usize,
    pub log: RingBuffer<String>,
}

impl Kernel {
    /// Boot the kernel: initialize all subsystems.
    pub fn boot(total_memory: usize, scheduler_algo: SchedulingAlgorithm) -> Self {
        let mut frame_allocator = PageFrameAllocator::new(total_memory);
        // Reserve the first 1 MB for the kernel image and BIOS.
        frame_allocator.reserve_range(0, 256); // 256 frames = 1 MB

        let mut kernel = Self {
            scheduler: Scheduler::new(scheduler_algo),
            frame_allocator,
            vfs: VFS::new(),
            address_spaces: HashMap::new(),
            pipes: HashMap::new(),
            message_queues: HashMap::new(),
            next_pipe_id: 1,
            next_mq_id: 1,
            log: RingBuffer::new(1024),
        };

        // Create the initial filesystem structure.
        kernel.vfs.mkdir("/bin", 0o755).ok();
        kernel.vfs.mkdir("/etc", 0o755).ok();
        kernel.vfs.mkdir("/home", 0o755).ok();
        kernel.vfs.mkdir("/tmp", 0o1777).ok();
        kernel.vfs.mkdir("/proc", 0o555).ok();
        kernel.vfs.mkdir("/dev", 0o755).ok();

        kernel.log.push("Kernel booted successfully".to_string());
        kernel.log.push(format!("Total memory: {} MB", total_memory / 1024 / 1024));
        kernel.log.push(format!("Free memory: {} MB", kernel.frame_allocator.free_memory() / 1024 / 1024));

        kernel
    }

    /// Handle a system call from a user process.
    pub fn syscall(&mut self, call: Syscall, args: &[u64]) -> SyscallResult {
        match call {
            Syscall::Getpid => {
                SyscallResult::Ok(self.scheduler.current_pid.unwrap_or(0) as i64)
            }
            Syscall::Getppid => {
                let pid = self.scheduler.current_pid.unwrap_or(0);
                let ppid = self.scheduler.processes.get(&pid)
                    .and_then(|p| p.parent_pid)
                    .unwrap_or(0);
                SyscallResult::Ok(ppid as i64)
            }
            Syscall::Fork => {
                let parent_pid = self.scheduler.current_pid.unwrap_or(0);
                let parent_name = self.scheduler.processes.get(&parent_pid)
                    .map(|p| p.name.clone())
                    .unwrap_or_default();
                let child_pid = self.scheduler.create_process(
                    &format!("{} (fork)", parent_name),
                    Some(parent_pid), 0,
                );
                self.log.push(format!("fork: {} -> {}", parent_pid, child_pid));
                SyscallResult::Ok(child_pid as i64)
            }
            Syscall::Exit => {
                let code = args.get(0).copied().unwrap_or(0) as i32;
                let pid = self.scheduler.current_pid.unwrap_or(0);
                self.scheduler.terminate_process(pid, code);
                self.log.push(format!("exit: pid {} with code {}", pid, code));
                SyscallResult::Ok(0)
            }
            Syscall::Kill => {
                let target_pid = args.get(0).copied().unwrap_or(0) as Pid;
                let signal_num = args.get(1).copied().unwrap_or(15) as u8;
                if let Some(proc) = self.scheduler.processes.get_mut(&target_pid) {
                    // Map signal number to Signal enum (simplified).
                    let signal = match signal_num {
                        2 => Signal::SIGINT,
                        9 => Signal::SIGKILL,
                        15 => Signal::SIGTERM,
                        _ => Signal::SIGTERM,
                    };
                    proc.send_signal(signal);
                    if signal == Signal::SIGKILL {
                        self.scheduler.terminate_process(target_pid, -9);
                    }
                    SyscallResult::Ok(0)
                } else {
                    SyscallResult::Err(-3) // ESRCH: No such process
                }
            }
            Syscall::Pipe => {
                let pipe_id = self.next_pipe_id;
                self.next_pipe_id += 1;
                self.pipes.insert(pipe_id, Pipe::new(pipe_id, 65536)); // 64KB buffer

                // Allocate two FDs in the current process: one for reading, one for writing.
                if let Some(pid) = self.scheduler.current_pid {
                    if let Some(proc) = self.scheduler.processes.get_mut(&pid) {
                        let read_fd = proc.alloc_fd(FileDescriptor {
                            kind: FdKind::Pipe { pipe_id },
                            flags: 0, offset: 0, inode: None,
                        });
                        let write_fd = proc.alloc_fd(FileDescriptor {
                            kind: FdKind::Pipe { pipe_id },
                            flags: 1, offset: 0, inode: None,
                        });
                        return SyscallResult::Ok(((read_fd as i64) << 32) | (write_fd as i64));
                    }
                }
                SyscallResult::Err(-9) // EBADF
            }
            Syscall::Nanosleep => {
                let nanos = args.get(0).copied().unwrap_or(0);
                self.scheduler.sleep_current(Duration::from_nanos(nanos));
                SyscallResult::Ok(0)
            }
            _ => SyscallResult::Err(-38), // ENOSYS: Function not implemented
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_frame_allocator() {
        let mut alloc = PageFrameAllocator::new(16 * PAGE_SIZE); // 16 frames = 64 KB
        assert_eq!(alloc.free_frames(), 16);

        let f1 = alloc.allocate_frame().unwrap();
        let f2 = alloc.allocate_frame().unwrap();
        assert_ne!(f1, f2);
        assert_eq!(alloc.free_frames(), 14);

        alloc.free_frame(f1);
        assert_eq!(alloc.free_frames(), 15);

        // The freed frame should be reusable.
        let f3 = alloc.allocate_frame().unwrap();
        assert_eq!(f3, f1); // Came from the free cache
    }

    #[test]
    fn test_contiguous_allocation() {
        let mut alloc = PageFrameAllocator::new(64 * PAGE_SIZE);
        let block = alloc.allocate_contiguous(8).unwrap();
        assert_eq!(alloc.free_frames(), 56);

        alloc.free_contiguous(block, 8);
        assert_eq!(alloc.free_frames(), 64);
    }

    #[test]
    fn test_virtual_memory_translation() {
        let mut vm = VirtualMemorySpace::new(1);
        vm.map_page(10, 42, PageFlags::user_rw()).unwrap();

        // Virtual address 10*4096 + 100 should translate to physical 42*4096 + 100.
        let phys = vm.translate(10 * PAGE_SIZE + 100, false, false, true).unwrap();
        assert_eq!(phys, 42 * PAGE_SIZE + 100);
    }

    #[test]
    fn test_page_fault_on_unmapped() {
        let mut vm = VirtualMemorySpace::new(1);
        let result = vm.translate(0x1000, false, false, true);
        assert!(matches!(result, Err(VmError::PageFault { reason: PageFaultReason::NotPresent, .. })));
    }

    #[test]
    fn test_write_protection() {
        let mut vm = VirtualMemorySpace::new(1);
        vm.map_page(5, 10, PageFlags::user_ro()).unwrap(); // Read-only page

        // Read should succeed.
        assert!(vm.translate(5 * PAGE_SIZE, false, false, true).is_ok());

        // Write should cause a page fault.
        let result = vm.translate(5 * PAGE_SIZE, true, false, true);
        assert!(matches!(result, Err(VmError::PageFault { reason: PageFaultReason::WriteToReadOnly, .. })));
    }

    #[test]
    fn test_buddy_allocator() {
        let mut buddy = BuddyAllocator::new(0, 1024, 64); // 1KB, 64-byte blocks

        let a1 = buddy.allocate(100).unwrap(); // Gets a 128-byte block
        let a2 = buddy.allocate(200).unwrap(); // Gets a 256-byte block
        assert_ne!(a1, a2);
        assert!(buddy.bytes_used() > 0);

        buddy.free(a1);
        buddy.free(a2);
        // After freeing, blocks should merge back.
        assert_eq!(buddy.bytes_used(), 0);
    }

    #[test]
    fn test_buddy_merge() {
        let mut buddy = BuddyAllocator::new(0, 256, 64); // 4 blocks of 64 bytes

        // Allocate all four 64-byte blocks.
        let a = buddy.allocate(64).unwrap();
        let b = buddy.allocate(64).unwrap();
        let c = buddy.allocate(64).unwrap();
        let d = buddy.allocate(64).unwrap();
        assert!(buddy.allocate(64).is_none()); // Full!

        // Free them in an order that tests merging.
        buddy.free(b);
        buddy.free(a); // a and b should merge into a 128-byte block
        buddy.free(d);
        buddy.free(c); // c and d merge, then the two 128-byte blocks merge into 256

        // Now we should be able to allocate the full 256 bytes.
        assert!(buddy.allocate(256).is_some());
    }

    #[test]
    fn test_slab_allocator() {
        let mut buddy = BuddyAllocator::new(0, 4096, 64);
        let mut slab = SlabCache::new("test_object", 48, 16);

        let addr1 = slab.allocate(&mut buddy).unwrap();
        let addr2 = slab.allocate(&mut buddy).unwrap();
        assert_ne!(addr1, addr2);
        assert_eq!(slab.active_objects(), 2);

        slab.free(addr1);
        assert_eq!(slab.active_objects(), 1);

        // Reallocating should reuse the freed slot.
        let addr3 = slab.allocate(&mut buddy).unwrap();
        assert_eq!(addr3, addr1);
    }

    #[test]
    fn test_scheduler_round_robin() {
        let mut sched = Scheduler::new(SchedulingAlgorithm::RoundRobin {
            quantum: Duration::from_millis(10),
        });

        let p1 = sched.create_process("process_a", None, 0);
        let p2 = sched.create_process("process_b", None, 0);
        let p3 = sched.create_process("process_c", None, 0);

        // First schedule should pick p1 (first in queue).
        let next = sched.schedule().unwrap();
        assert_eq!(next, p1);

        // Second schedule: p1 goes to back of queue, p2 runs.
        let next = sched.schedule().unwrap();
        assert_eq!(next, p2);

        // Third: p3.
        let next = sched.schedule().unwrap();
        assert_eq!(next, p3);

        // Fourth: back to p1 (round-robin).
        let next = sched.schedule().unwrap();
        assert_eq!(next, p1);
    }

    #[test]
    fn test_scheduler_priority() {
        let mut sched = Scheduler::new(SchedulingAlgorithm::Priority);

        let low = sched.create_process("low_priority", None, 10);
        let high = sched.create_process("high_priority", None, -10);
        let mid = sched.create_process("mid_priority", None, 0);

        // Highest priority (most negative nice value) should run first.
        let next = sched.schedule().unwrap();
        assert_eq!(next, high);
    }

    #[test]
    fn test_vfs_basic_operations() {
        let mut vfs = VFS::new();

        // Create a directory structure.
        vfs.mkdir("/home", 0o755).unwrap();
        vfs.mkdir("/home/user", 0o755).unwrap();

        // Create a file.
        vfs.create_file("/home/user/hello.txt", 0o644, b"Hello, World!".to_vec()).unwrap();

        // Read it back.
        let content = vfs.read_file("/home/user/hello.txt").unwrap();
        assert_eq!(content, b"Hello, World!");

        // List directory.
        let entries = vfs.readdir("/home/user").unwrap();
        let names: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
        assert!(names.contains(&"hello.txt".to_string()));
        assert!(names.contains(&".".to_string()));
        assert!(names.contains(&"..".to_string()));

        // Check stat.
        let stat = vfs.stat("/home/user/hello.txt").unwrap();
        assert_eq!(stat.size, 13);
        assert_eq!(stat.permissions.to_mode(), 0o644);
    }

    #[test]
    fn test_vfs_write_and_delete() {
        let mut vfs = VFS::new();
        vfs.create_file("/test.txt", 0o644, b"original".to_vec()).unwrap();

        // Overwrite.
        vfs.write_file("/test.txt", b"modified".to_vec()).unwrap();
        assert_eq!(vfs.read_file("/test.txt").unwrap(), b"modified");

        // Delete.
        vfs.unlink("/test.txt").unwrap();
        assert!(vfs.read_file("/test.txt").is_err());
    }

    #[test]
    fn test_pipe_ipc() {
        let mut pipe = Pipe::new(1, 1024);

        // Write some data.
        let written = pipe.write(b"Hello from process A!").unwrap();
        assert_eq!(written, 21);
        assert_eq!(pipe.bytes_available(), 21);

        // Read it from the other end.
        let mut buf = vec![0u8; 100];
        let read = pipe.read(&mut buf).unwrap();
        assert_eq!(read, 21);
        assert_eq!(&buf[..21], b"Hello from process A!");
    }

    #[test]
    fn test_message_queue() {
        let mut mq = MessageQueue::new(1, 10);

        mq.send(Message { msg_type: 1, data: b"hello".to_vec(), sender_pid: 100 }).unwrap();
        mq.send(Message { msg_type: 2, data: b"world".to_vec(), sender_pid: 101 }).unwrap();

        // Receive any message.
        let msg = mq.receive(0).unwrap();
        assert_eq!(msg.data, b"hello");

        // Receive by type.
        let msg = mq.receive(2).unwrap();
        assert_eq!(msg.data, b"world");
    }

    #[test]
    fn test_ring_buffer() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(4);

        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        assert!(rb.push(4));
        assert!(!rb.push(5)); // Full!

        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert!(rb.push(5)); // Space freed up.
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
        assert_eq!(rb.pop(), Some(5));
        assert_eq!(rb.pop(), None); // Empty!
    }

    #[test]
    fn test_kernel_boot_and_syscalls() {
        let mut kernel = Kernel::boot(64 * 1024 * 1024, // 64 MB RAM
            SchedulingAlgorithm::RoundRobin { quantum: Duration::from_millis(10) });

        // Create a process.
        let init_pid = kernel.scheduler.create_process("init", None, 0);
        kernel.scheduler.schedule(); // Start running init.

        // Test getpid syscall.
        if let SyscallResult::Ok(pid) = kernel.syscall(Syscall::Getpid, &[]) {
            assert_eq!(pid, init_pid as i64);
        }

        // Test fork syscall.
        if let SyscallResult::Ok(child_pid) = kernel.syscall(Syscall::Fork, &[]) {
            assert!(child_pid > init_pid as i64);
        }

        // Test pipe syscall.
        if let SyscallResult::Ok(fds) = kernel.syscall(Syscall::Pipe, &[]) {
            let read_fd = (fds >> 32) as usize;
            let write_fd = (fds & 0xFFFFFFFF) as usize;
            assert_ne!(read_fd, write_fd);
        }

        // Test kill syscall (kill the child).
        let child_pid = 2; // From the fork above
        let result = kernel.syscall(Syscall::Kill, &[child_pid as u64, 9]); // SIGKILL
        assert!(matches!(result, SyscallResult::Ok(0)));

        // Verify the filesystem was initialized.
        assert!(kernel.vfs.stat("/home").is_ok());
        assert!(kernel.vfs.stat("/tmp").is_ok());
        assert!(kernel.vfs.stat("/proc").is_ok());
    }

    #[test]
    fn test_permissions() {
        let perm = Permissions::from_mode(0o755);
        assert!(perm.owner_read && perm.owner_write && perm.owner_execute);
        assert!(perm.group_read && !perm.group_write && perm.group_execute);
        assert!(perm.other_read && !perm.other_write && perm.other_execute);
        assert_eq!(perm.to_mode(), 0o755);
        assert_eq!(perm.to_string(), "rwxr-xr-x");

        let perm2 = Permissions::from_mode(0o4644);
        assert!(perm2.setuid);
        assert_eq!(perm2.to_string(), "rwSr--r--");
    }
}
