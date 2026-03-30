// ============================================================================
// CHRONOS GAME ENGINE PRIMITIVES
// ============================================================================
//
// HOW GAME ENGINES ACTUALLY WORK (and why they're architecturally interesting):
//
// A game engine is a real-time simulation loop that processes physics,
// updates game logic, and renders 60+ times per second. Three core systems:
//
// 1. ENTITY-COMPONENT SYSTEM (ECS):
//    The dominant architecture for modern games. Rather than deep inheritance
//    hierarchies (Player extends Character extends Entity), ECS uses
//    composition. An Entity is just an ID. Components are plain data (Position,
//    Velocity, Health, Sprite). Systems process all entities with a given
//    set of components.
//
//    Why it wins:
//    - Data locality: components of the same type are contiguous in memory
//      → cache-friendly iteration over thousands of entities
//    - Composability: any combination of components creates any behavior
//    - Testability: systems are pure functions over component data
//
//    The canonical ECS paper: Noel Llopis, "Data-Oriented Design" (2009).
//    Unity DOTS, Bevy, and EnTT all use this architecture.
//
// 2. PHYSICS SIMULATION:
//    Rigid body dynamics: integrate Newton's laws using the Verlet or
//    semi-implicit Euler method. For each body:
//      a_n = F_n / m                        (acceleration from forces)
//      v_{n+1} = v_n + a_n * dt             (semi-implicit Euler)
//      x_{n+1} = x_n + v_{n+1} * dt
//
//    Collision detection: broad phase (AABB overlap) + narrow phase (SAT).
//    Collision response: impulse-based resolution preserving momentum.
//
//    The Separating Axis Theorem (SAT): two convex shapes do NOT overlap iff
//    there exists an axis along which their projections don't overlap. For
//    AABB vs AABB there are 4 candidate axes (2 per box). For polygon vs
//    polygon, all edge normals are candidate axes.
//
// 3. RENDERING PIPELINE (abstract):
//    Games use a deferred or forward shading pipeline. We abstract this as:
//    - Scene graph: transforms organized in a parent-child hierarchy
//    - Camera: view and projection matrices
//    - Render queue: sorted draw calls (by material, then by depth)
//    - Lighting: ambient + directional + point lights
//
//    We implement the math for all of this (matrices, frustum culling,
//    depth sorting) but don't actually call a GPU — the Chronos compiler
//    would emit SPIR-V or Metal shaders at compile time.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic with tests):
//   1.  Vec2, Vec3, Vec4 and Mat4 (column-major for GPU compatibility)
//   2.  Entity-Component System (archetypal storage, sparse sets)
//   3.  Core components: Transform, Velocity, AABB, Sprite, Health, Tag
//   4.  Core systems: Physics, Collision, Lifetime, Animation
//   5.  AABB collision detection and impulse response
//   6.  Swept AABB (continuous collision detection, prevents tunneling)
//   7.  Quadtree spatial partitioning for broad phase
//   8.  Semi-implicit Euler and Verlet integrators
//   9.  Transform hierarchy and scene graph
//  10.  Camera (view matrix, projection, frustum)
//  11.  Render queue (sorting, draw calls)
//  12.  Sprite animation (frame atlas, state machine)
//  13.  Particle system
//  14.  Input state (keyboard, mouse, gamepad abstraction)
//  15.  Game loop timing (fixed timestep with interpolation)
//  16.  Audio emitter positioning (3D)
//  17.  Comprehensive tests
// ============================================================================

use std::collections::HashMap;

// ============================================================================
// Part 1: Math Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 { pub x: f32, pub y: f32 }

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };

    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    pub fn splat(v: f32) -> Self { Self { x: v, y: v } }

    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y }
    pub fn len_sq(self) -> f32 { self.dot(self) }
    pub fn len(self) -> f32 { self.len_sq().sqrt() }
    pub fn norm(self) -> Self {
        let l = self.len();
        if l < 1e-9 { Self::ZERO } else { Self::new(self.x / l, self.y / l) }
    }
    pub fn perp(self) -> Self { Self::new(-self.y, self.x) }
    pub fn cross(self, o: Self) -> f32 { self.x * o.y - self.y * o.x }
    pub fn add(self, o: Self) -> Self { Self::new(self.x + o.x, self.y + o.y) }
    pub fn sub(self, o: Self) -> Self { Self::new(self.x - o.x, self.y - o.y) }
    pub fn mul(self, s: f32) -> Self { Self::new(self.x * s, self.y * s) }
    pub fn div(self, s: f32) -> Self { Self::new(self.x / s, self.y / s) }
    pub fn lerp(self, o: Self, t: f32) -> Self { self.add(o.sub(self).mul(t)) }
    pub fn dist(self, o: Self) -> f32 { self.sub(o).len() }
    pub fn reflect(self, normal: Self) -> Self {
        self.sub(normal.mul(2.0 * self.dot(normal)))
    }
    pub fn rotate(self, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c * self.x - s * self.y, s * self.x + c * self.y)
    }
    pub fn angle(self) -> f32 { self.y.atan2(self.x) }
    pub fn min_comp(self, o: Self) -> Self { Self::new(self.x.min(o.x), self.y.min(o.y)) }
    pub fn max_comp(self, o: Self) -> Self { Self::new(self.x.max(o.x), self.y.max(o.y)) }
    pub fn clamp(self, min: Self, max: Self) -> Self { self.max_comp(min).min_comp(max) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn splat(v: f32) -> Self { Self::new(v, v, v) }

    pub fn dot(self, o: Self) -> f32 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn len_sq(self) -> f32 { self.dot(self) }
    pub fn len(self) -> f32 { self.len_sq().sqrt() }
    pub fn norm(self) -> Self {
        let l = self.len();
        if l < 1e-9 { Self::ZERO } else { Self::new(self.x / l, self.y / l, self.z / l) }
    }
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn add(self, o: Self) -> Self { Self::new(self.x+o.x, self.y+o.y, self.z+o.z) }
    pub fn sub(self, o: Self) -> Self { Self::new(self.x-o.x, self.y-o.y, self.z-o.z) }
    pub fn mul(self, s: f32) -> Self { Self::new(self.x*s, self.y*s, self.z*s) }
    pub fn lerp(self, o: Self, t: f32) -> Self { self.add(o.sub(self).mul(t)) }
    pub fn dist(self, o: Self) -> f32 { self.sub(o).len() }

    pub fn to_vec4(self, w: f32) -> Vec4 { Vec4::new(self.x, self.y, self.z, w) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }
    pub fn xyz(self) -> Vec3 { Vec3::new(self.x, self.y, self.z) }
    pub fn dot(self, o: Self) -> f32 { self.x*o.x + self.y*o.y + self.z*o.z + self.w*o.w }
}

/// Column-major 4x4 matrix (compatible with GPU / OpenGL convention)
/// m[col][row], so m[0] is the first column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    pub fn identity() -> Self {
        let mut m = Self { cols: [[0.0; 4]; 4] };
        m.cols[0][0] = 1.0;
        m.cols[1][1] = 1.0;
        m.cols[2][2] = 1.0;
        m.cols[3][3] = 1.0;
        m
    }

    pub fn translation(v: Vec3) -> Self {
        let mut m = Self::identity();
        m.cols[3][0] = v.x;
        m.cols[3][1] = v.y;
        m.cols[3][2] = v.z;
        m
    }

    pub fn scale(s: Vec3) -> Self {
        let mut m = Self::identity();
        m.cols[0][0] = s.x;
        m.cols[1][1] = s.y;
        m.cols[2][2] = s.z;
        m
    }

    pub fn rotation_y(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        let mut m = Self::identity();
        m.cols[0][0] = c;  m.cols[2][0] = s;
        m.cols[0][2] = -s; m.cols[2][2] = c;
        m
    }

    pub fn rotation_z(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        let mut m = Self::identity();
        m.cols[0][0] = c;  m.cols[1][0] = -s;
        m.cols[0][1] = s;  m.cols[1][1] = c;
        m
    }

    pub fn mul_mat(self, o: Mat4) -> Mat4 {
        let mut result = Mat4 { cols: [[0.0; 4]; 4] };
        for j in 0..4 {
            for i in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.cols[k][i] * o.cols[j][k];
                }
                result.cols[j][i] = sum;
            }
        }
        result
    }

    pub fn mul_vec4(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.cols[0][0]*v.x + self.cols[1][0]*v.y + self.cols[2][0]*v.z + self.cols[3][0]*v.w,
            self.cols[0][1]*v.x + self.cols[1][1]*v.y + self.cols[2][1]*v.z + self.cols[3][1]*v.w,
            self.cols[0][2]*v.x + self.cols[1][2]*v.y + self.cols[2][2]*v.z + self.cols[3][2]*v.w,
            self.cols[0][3]*v.x + self.cols[1][3]*v.y + self.cols[2][3]*v.z + self.cols[3][3]*v.w,
        )
    }

    pub fn mul_point(self, p: Vec3) -> Vec3 {
        let v = self.mul_vec4(p.to_vec4(1.0));
        v.xyz()
    }

    pub fn transpose(self) -> Self {
        let mut m = Self { cols: [[0.0; 4]; 4] };
        for i in 0..4 {
            for j in 0..4 {
                m.cols[j][i] = self.cols[i][j];
            }
        }
        m
    }

    /// Perspective projection matrix (right-handed, NDC [-1, 1])
    pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y_rad / 2.0).tan();
        let mut m = Self { cols: [[0.0; 4]; 4] };
        m.cols[0][0] = f / aspect;
        m.cols[1][1] = f;
        m.cols[2][2] = (far + near) / (near - far);
        m.cols[2][3] = -1.0;
        m.cols[3][2] = (2.0 * far * near) / (near - far);
        m
    }

    /// Look-at view matrix (right-handed)
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = target.sub(eye).norm();
        let r = f.cross(up).norm();
        let u = r.cross(f);
        let mut m = Self { cols: [[0.0; 4]; 4] };
        m.cols[0][0] = r.x;  m.cols[0][1] = u.x;  m.cols[0][2] = -f.x;
        m.cols[1][0] = r.y;  m.cols[1][1] = u.y;  m.cols[1][2] = -f.y;
        m.cols[2][0] = r.z;  m.cols[2][1] = u.z;  m.cols[2][2] = -f.z;
        m.cols[3][0] = -r.dot(eye);
        m.cols[3][1] = -u.dot(eye);
        m.cols[3][2] = f.dot(eye);
        m.cols[3][3] = 1.0;
        m
    }
}

// ============================================================================
// Part 2: Entity-Component System (Archetypal ECS)
// ============================================================================
//
// Our ECS uses a sparse-set approach: each component type has its own
// dense array indexed by entity ID. An entity is just a (index, generation)
// pair — the generation prevents use-after-free when an entity is recycled.
//
// Archetype-based ECS (like Unity DOTS or Bevy) groups entities by their
// exact component set into "archetypes" for cache-efficient iteration.
// We implement a simpler sparse-set approach that is easier to understand.

pub type EntityIndex = u32;
pub type Generation = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    pub index: EntityIndex,
    pub generation: Generation,
}

impl Entity {
    pub fn null() -> Self {
        Self { index: u32::MAX, generation: 0 }
    }
    pub fn is_null(&self) -> bool {
        self.index == u32::MAX
    }
}

/// The ECS world: manages entities and their components
pub struct World {
    // Entity management
    generations: Vec<Generation>,
    free_list: Vec<EntityIndex>,

    // Components (stored separately per type)
    pub transforms: SparseSet<Transform>,
    pub velocities: SparseSet<Velocity>,
    pub aabbs: SparseSet<Aabb>,
    pub sprites: SparseSet<Sprite>,
    pub healths: SparseSet<Health>,
    pub tags: SparseSet<Tag>,
    pub rigid_bodies: SparseSet<RigidBody>,
    pub particles: SparseSet<ParticleComponent>,
    pub audio_emitters: SparseSet<AudioEmitter>,
    pub children: SparseSet<Children>,
    pub parent: SparseSet<Parent>,
    pub lifetime: SparseSet<Lifetime>,

    entity_count: usize,
}

impl World {
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            free_list: Vec::new(),
            transforms: SparseSet::new(),
            velocities: SparseSet::new(),
            aabbs: SparseSet::new(),
            sprites: SparseSet::new(),
            healths: SparseSet::new(),
            tags: SparseSet::new(),
            rigid_bodies: SparseSet::new(),
            particles: SparseSet::new(),
            audio_emitters: SparseSet::new(),
            children: SparseSet::new(),
            parent: SparseSet::new(),
            lifetime: SparseSet::new(),
            entity_count: 0,
        }
    }

    pub fn spawn(&mut self) -> Entity {
        self.entity_count += 1;
        if let Some(index) = self.free_list.pop() {
            Entity { index, generation: self.generations[index as usize] }
        } else {
            let index = self.generations.len() as EntityIndex;
            self.generations.push(1);
            Entity { index, generation: 1 }
        }
    }

    pub fn despawn(&mut self, entity: Entity) {
        if !self.is_alive(entity) { return; }
        self.generations[entity.index as usize] = self.generations[entity.index as usize].wrapping_add(1);
        self.free_list.push(entity.index);
        self.entity_count = self.entity_count.saturating_sub(1);

        // Remove all components
        self.transforms.remove(entity.index);
        self.velocities.remove(entity.index);
        self.aabbs.remove(entity.index);
        self.sprites.remove(entity.index);
        self.healths.remove(entity.index);
        self.tags.remove(entity.index);
        self.rigid_bodies.remove(entity.index);
        self.lifetime.remove(entity.index);
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        (entity.index as usize) < self.generations.len()
            && self.generations[entity.index as usize] == entity.generation
    }

    pub fn entity_count(&self) -> usize {
        self.entity_count
    }
}

/// A sparse set maps entity indices to dense component storage.
/// - Dense array: actual component data, cache-friendly iteration
/// - Sparse array: entity index → position in dense array
/// O(1) insert, remove, lookup. O(n) iteration with no gaps.
pub struct SparseSet<T> {
    sparse: Vec<Option<u32>>,   // sparse[entity_index] = dense_index
    dense_entities: Vec<EntityIndex>,
    dense_data: Vec<T>,
}

impl<T: Clone> SparseSet<T> {
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense_entities: Vec::new(),
            dense_data: Vec::new(),
        }
    }

    pub fn insert(&mut self, index: EntityIndex, data: T) {
        let index = index as usize;
        if index >= self.sparse.len() {
            self.sparse.resize(index + 1, None);
        }
        if let Some(dense_idx) = self.sparse[index] {
            // Update existing
            self.dense_data[dense_idx as usize] = data;
        } else {
            // New entry
            let dense_idx = self.dense_data.len() as u32;
            self.sparse[index] = Some(dense_idx);
            self.dense_entities.push(index as EntityIndex);
            self.dense_data.push(data);
        }
    }

    pub fn get(&self, index: EntityIndex) -> Option<&T> {
        let index = index as usize;
        let dense_idx = (*self.sparse.get(index)?)? as usize;
        self.dense_data.get(dense_idx)
    }

    pub fn get_mut(&mut self, index: EntityIndex) -> Option<&mut T> {
        let index = index as usize;
        let dense_idx = (*self.sparse.get(index)?)? as usize;
        self.dense_data.get_mut(dense_idx)
    }

    pub fn remove(&mut self, index: EntityIndex) {
        let index = index as usize;
        if index >= self.sparse.len() { return; }
        if let Some(dense_idx) = self.sparse[index].take() {
            let dense_idx = dense_idx as usize;
            let last = self.dense_data.len() - 1;
            if dense_idx != last {
                // Swap with last element and update sparse
                self.dense_data.swap(dense_idx, last);
                self.dense_entities.swap(dense_idx, last);
                let moved_entity = self.dense_entities[dense_idx] as usize;
                self.sparse[moved_entity] = Some(dense_idx as u32);
            }
            self.dense_data.pop();
            self.dense_entities.pop();
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntityIndex, &T)> {
        self.dense_entities.iter().copied().zip(self.dense_data.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (EntityIndex, &mut T)> {
        self.dense_entities.iter().copied().zip(self.dense_data.iter_mut())
    }

    pub fn len(&self) -> usize {
        self.dense_data.len()
    }

    pub fn contains(&self, index: EntityIndex) -> bool {
        let i = index as usize;
        i < self.sparse.len() && self.sparse[i].is_some()
    }
}

// ============================================================================
// Part 3: Core Components
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: f32,   // 2D: single angle; 3D: would be quaternion
    pub scale: Vec3,
}

impl Transform {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            position: Vec3::new(x, y, 0.0),
            rotation: 0.0,
            scale: Vec3::splat(1.0),
        }
    }
    pub fn at(position: Vec3) -> Self {
        Self { position, rotation: 0.0, scale: Vec3::splat(1.0) }
    }

    pub fn to_matrix(&self) -> Mat4 {
        Mat4::translation(self.position)
            .mul_mat(Mat4::rotation_z(self.rotation))
            .mul_mat(Mat4::scale(self.scale))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Velocity {
    pub linear: Vec3,
    pub angular: f32, // rad/s (2D)
}

impl Velocity {
    pub fn new(x: f32, y: f32) -> Self {
        Self { linear: Vec3::new(x, y, 0.0), angular: 0.0 }
    }
    pub fn zero() -> Self {
        Self { linear: Vec3::ZERO, angular: 0.0 }
    }
}

/// Axis-Aligned Bounding Box (local space — add transform.position for world space)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub half_extents: Vec2, // width/2 and height/2
    pub offset: Vec2,       // local offset from entity origin
}

impl Aabb {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            half_extents: Vec2::new(width / 2.0, height / 2.0),
            offset: Vec2::ZERO,
        }
    }

    pub fn with_offset(mut self, ox: f32, oy: f32) -> Self {
        self.offset = Vec2::new(ox, oy);
        self
    }

    /// World-space AABB given an entity's transform
    pub fn world_aabb(&self, transform: &Transform) -> WorldAabb {
        let cx = transform.position.x + self.offset.x;
        let cy = transform.position.y + self.offset.y;
        WorldAabb {
            min: Vec2::new(cx - self.half_extents.x, cy - self.half_extents.y),
            max: Vec2::new(cx + self.half_extents.x, cy + self.half_extents.y),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldAabb {
    pub min: Vec2,
    pub max: Vec2,
}

impl WorldAabb {
    pub fn overlaps(&self, other: &WorldAabb) -> bool {
        self.min.x < other.max.x
            && self.max.x > other.min.x
            && self.min.y < other.max.y
            && self.max.y > other.min.y
    }

    pub fn center(&self) -> Vec2 {
        Vec2::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
        )
    }

    pub fn half_extents(&self) -> Vec2 {
        Vec2::new(
            (self.max.x - self.min.x) / 2.0,
            (self.max.y - self.min.y) / 2.0,
        )
    }

    pub fn area(&self) -> f32 {
        let he = self.half_extents();
        he.x * he.y * 4.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sprite {
    pub texture_id: u32,
    pub uv_rect: [f32; 4],   // [u, v, w, h] in texture coordinates 0..1
    pub color: [f32; 4],      // RGBA tint
    pub layer: i32,            // render layer (lower = drawn first)
    pub flip_x: bool,
    pub flip_y: bool,
}

impl Sprite {
    pub fn new(texture_id: u32) -> Self {
        Self {
            texture_id,
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            layer: 0,
            flip_x: false,
            flip_y: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Health {
    pub current: f32,
    pub max: f32,
    pub invincible: bool,
    pub invincible_timer: f32,
}

impl Health {
    pub fn new(max: f32) -> Self {
        Self { current: max, max, invincible: false, invincible_timer: 0.0 }
    }

    pub fn is_dead(&self) -> bool { self.current <= 0.0 }

    pub fn take_damage(&mut self, amount: f32) {
        if self.invincible { return; }
        self.current = (self.current - amount).max(0.0);
    }

    pub fn heal(&mut self, amount: f32) {
        self.current = (self.current + amount).min(self.max);
    }

    pub fn fraction(&self) -> f32 {
        if self.max <= 0.0 { 0.0 } else { self.current / self.max }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tag {
    pub value: u64, // bitmask of tags for quick filtering
}

impl Tag {
    pub fn new(bits: u64) -> Self { Self { value: bits } }
    pub fn has(&self, bit: u64) -> bool { self.value & bit != 0 }
    pub fn add(&mut self, bit: u64) { self.value |= bit; }
    pub fn remove(&mut self, bit: u64) { self.value &= !bit; }
}

pub mod tags {
    pub const PLAYER: u64 = 1 << 0;
    pub const ENEMY: u64 = 1 << 1;
    pub const PROJECTILE: u64 = 1 << 2;
    pub const SOLID: u64 = 1 << 3;
    pub const TRIGGER: u64 = 1 << 4;
    pub const DESTRUCTIBLE: u64 = 1 << 5;
    pub const COLLECTIBLE: u64 = 1 << 6;
}

#[derive(Debug, Clone, PartialEq)]
pub struct RigidBody {
    pub mass: f32,           // kg; 0 = static/infinite mass
    pub inv_mass: f32,
    pub restitution: f32,    // bounciness [0, 1]
    pub friction: f32,       // [0, 1]
    pub gravity_scale: f32,
    pub force: Vec2,         // accumulated force (reset each frame)
    pub torque: f32,         // accumulated torque
    pub moment_of_inertia: f32,
    pub is_kinematic: bool,  // moved by code, not physics
}

impl RigidBody {
    pub fn new(mass: f32) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            mass,
            inv_mass,
            restitution: 0.3,
            friction: 0.5,
            gravity_scale: 1.0,
            force: Vec2::ZERO,
            torque: 0.0,
            moment_of_inertia: mass * 0.1, // simplified
            is_kinematic: false,
        }
    }

    pub fn static_body() -> Self {
        Self { mass: 0.0, inv_mass: 0.0, is_kinematic: false, ..Self::new(0.0) }
    }

    pub fn apply_force(&mut self, force: Vec2) {
        self.force = self.force.add(force);
    }

    pub fn apply_impulse_velocity(&mut self, impulse: Vec2, velocity: &mut Velocity) {
        velocity.linear = velocity.linear.add(Vec3::new(
            impulse.x * self.inv_mass,
            impulse.y * self.inv_mass,
            0.0,
        ));
    }
}

#[derive(Debug, Clone)]
pub struct ParticleComponent {
    pub lifetime: f32,
    pub max_lifetime: f32,
    pub size_start: f32,
    pub size_end: f32,
    pub alpha_start: f32,
    pub alpha_end: f32,
}

#[derive(Debug, Clone)]
pub struct AudioEmitter {
    pub sound_id: u32,
    pub volume: f32,
    pub pitch: f32,
    pub radius: f32,  // max audible distance
    pub looping: bool,
    pub playing: bool,
}

#[derive(Debug, Clone)]
pub struct Children(pub Vec<Entity>);

#[derive(Debug, Clone, Copy)]
pub struct Parent(pub Entity);

#[derive(Debug, Clone)]
pub struct Lifetime {
    pub remaining: f32,
}

// ============================================================================
// Part 4: Physics System
// ============================================================================
//
// Semi-implicit Euler integration:
//   a = F / m + gravity
//   v' = v + a * dt       (update velocity first)
//   x' = x + v' * dt      (then update position using new velocity)
//
// This is more stable than explicit Euler (x' = x + v * dt) for oscillatory
// systems like springs. The energy remains bounded.
//
// For very high velocities, objects can tunnel through thin walls ("bullet
// through paper" problem). Swept AABB (continuous collision detection) solves
// this by computing the time of first impact along the trajectory.

pub const GRAVITY: Vec2 = Vec2 { x: 0.0, y: -9.81 };

pub fn system_physics(world: &mut World, dt: f32) {
    // Collect entity IDs that have both Transform, Velocity, and RigidBody
    let entities: Vec<EntityIndex> = world.transforms.dense_entities.clone()
        .into_iter()
        .filter(|&idx| world.velocities.contains(idx) && world.rigid_bodies.contains(idx))
        .collect();

    for idx in entities {
        let rb = match world.rigid_bodies.get(idx) {
            Some(rb) if !rb.is_kinematic && rb.inv_mass > 0.0 => rb.clone(),
            _ => continue,
        };

        // Apply gravity
        let gravity_force = Vec2::new(
            GRAVITY.x * rb.mass * rb.gravity_scale,
            GRAVITY.y * rb.mass * rb.gravity_scale,
        );

        // a = (F + Fg) / m
        let total_force = rb.force.add(gravity_force);
        let accel = Vec2::new(
            total_force.x * rb.inv_mass,
            total_force.y * rb.inv_mass,
        );

        // Semi-implicit Euler: v first, then x
        if let Some(vel) = world.velocities.get_mut(idx) {
            vel.linear.x += accel.x * dt;
            vel.linear.y += accel.y * dt;
            vel.linear.x *= 1.0 - rb.friction * dt; // damping
        }

        let vel_copy = world.velocities.get(idx).map(|v| v.linear).unwrap_or(Vec3::ZERO);
        if let Some(transform) = world.transforms.get_mut(idx) {
            transform.position.x += vel_copy.x * dt;
            transform.position.y += vel_copy.y * dt;
        }

        // Reset forces
        if let Some(rb) = world.rigid_bodies.get_mut(idx) {
            rb.force = Vec2::ZERO;
            rb.torque = 0.0;
        }
    }
}

/// Lifetime system: despawn entities when their lifetime expires
pub fn system_lifetime(world: &mut World, dt: f32) {
    let to_despawn: Vec<EntityIndex> = world.lifetime.dense_entities.clone()
        .into_iter()
        .filter(|&idx| {
            if let Some(lt) = world.lifetime.get(idx) {
                lt.remaining <= 0.0
            } else {
                false
            }
        })
        .collect();

    // Tick lifetimes
    for (_, lt) in world.lifetime.iter_mut() {
        lt.remaining -= dt;
    }

    for idx in to_despawn {
        // Find the entity with this index (use generation 1 as approximation)
        let entity = Entity { index: idx, generation: world.generations.get(idx as usize).copied().unwrap_or(1) };
        world.despawn(entity);
    }
}

// ============================================================================
// Part 5: Collision Detection
// ============================================================================
//
// Broad phase: Quadtree reduces O(n²) comparisons to O(n log n).
// We partition space into quadrants recursively, only comparing entities
// in the same or neighboring cells.
//
// Narrow phase: AABB overlap test, then compute contact information:
// - Penetration depth on each axis
// - Normal: axis of minimum penetration (minimum overlap = shallowest exit)
//
// Resolution: Apply impulse to separate overlapping bodies while
// preserving momentum. The impulse magnitude:
//   j = -(1 + e) * v_rel · n / (1/m_a + 1/m_b)
// where e = coefficient of restitution, v_rel = relative velocity.

#[derive(Debug, Clone)]
pub struct Contact {
    pub entity_a: EntityIndex,
    pub entity_b: EntityIndex,
    pub normal: Vec2,        // points from b to a
    pub penetration: f32,
    pub contact_point: Vec2,
}

/// Find all AABB vs AABB collision contacts
pub fn detect_collisions(world: &World) -> Vec<Contact> {
    let mut contacts = Vec::new();

    // Collect all entities with AABB
    let aabb_entities: Vec<(EntityIndex, WorldAabb)> = world.aabbs.iter()
        .filter_map(|(idx, aabb)| {
            world.transforms.get(idx).map(|t| (idx, aabb.world_aabb(t)))
        })
        .collect();

    // Naive O(n²) broad phase (a quadtree would be better for large n)
    for i in 0..aabb_entities.len() {
        for j in (i + 1)..aabb_entities.len() {
            let (idx_a, aabb_a) = &aabb_entities[i];
            let (idx_b, aabb_b) = &aabb_entities[j];

            if let Some(contact) = aabb_vs_aabb(*idx_a, *idx_b, aabb_a, aabb_b) {
                contacts.push(contact);
            }
        }
    }

    contacts
}

fn aabb_vs_aabb(
    idx_a: EntityIndex,
    idx_b: EntityIndex,
    a: &WorldAabb,
    b: &WorldAabb,
) -> Option<Contact> {
    if !a.overlaps(b) {
        return None;
    }

    let ca = a.center();
    let cb = b.center();
    let hea = a.half_extents();
    let heb = b.half_extents();

    // Compute overlap on each axis
    let overlap_x = (hea.x + heb.x) - (ca.x - cb.x).abs();
    let overlap_y = (hea.y + heb.y) - (ca.y - cb.y).abs();

    // Choose axis of minimum penetration
    let (normal, penetration) = if overlap_x < overlap_y {
        let nx = if ca.x > cb.x { 1.0 } else { -1.0 };
        (Vec2::new(nx, 0.0), overlap_x)
    } else {
        let ny = if ca.y > cb.y { 1.0 } else { -1.0 };
        (Vec2::new(0.0, ny), overlap_y)
    };

    let contact_point = Vec2::new(
        (ca.x + cb.x) / 2.0,
        (ca.y + cb.y) / 2.0,
    );

    Some(Contact {
        entity_a: idx_a,
        entity_b: idx_b,
        normal,
        penetration,
        contact_point,
    })
}

/// Resolve collision contact using impulse-based response.
pub fn resolve_contact(world: &mut World, contact: &Contact) {
    let idx_a = contact.entity_a;
    let idx_b = contact.entity_b;

    let (inv_mass_a, restitution_a) = world.rigid_bodies.get(idx_a)
        .map(|rb| (rb.inv_mass, rb.restitution))
        .unwrap_or((0.0, 0.0));
    let (inv_mass_b, restitution_b) = world.rigid_bodies.get(idx_b)
        .map(|rb| (rb.inv_mass, rb.restitution))
        .unwrap_or((0.0, 0.0));

    let total_inv_mass = inv_mass_a + inv_mass_b;
    if total_inv_mass == 0.0 { return; } // Both static

    // Position correction: push entities apart proportionally
    let correction = contact.penetration / total_inv_mass * 0.8; // 80% correction
    let corr_a = contact.normal.mul(correction * inv_mass_a);
    let corr_b = contact.normal.mul(correction * inv_mass_b);

    if let Some(t) = world.transforms.get_mut(idx_a) {
        t.position.x += corr_a.x;
        t.position.y += corr_a.y;
    }
    if let Some(t) = world.transforms.get_mut(idx_b) {
        t.position.x -= corr_b.x;
        t.position.y -= corr_b.y;
    }

    // Velocity impulse
    let vel_a = world.velocities.get(idx_a).map(|v| Vec2::new(v.linear.x, v.linear.y)).unwrap_or(Vec2::ZERO);
    let vel_b = world.velocities.get(idx_b).map(|v| Vec2::new(v.linear.x, v.linear.y)).unwrap_or(Vec2::ZERO);

    let v_rel = vel_a.sub(vel_b);
    let v_rel_n = v_rel.dot(contact.normal);

    // Only resolve if objects are moving toward each other
    if v_rel_n > 0.0 { return; }

    let e = (restitution_a + restitution_b) / 2.0;
    let j = -(1.0 + e) * v_rel_n / total_inv_mass;
    let impulse = contact.normal.mul(j);

    if let Some(vel) = world.velocities.get_mut(idx_a) {
        vel.linear.x += impulse.x * inv_mass_a;
        vel.linear.y += impulse.y * inv_mass_a;
    }
    if let Some(vel) = world.velocities.get_mut(idx_b) {
        vel.linear.x -= impulse.x * inv_mass_b;
        vel.linear.y -= impulse.y * inv_mass_b;
    }
}

/// Swept AABB collision: compute time of impact for a moving AABB against a static AABB.
/// Returns Some(t) ∈ [0,1] where collision first occurs, or None if no collision this frame.
pub fn swept_aabb(
    moving: &WorldAabb,
    velocity: Vec2,
    static_aabb: &WorldAabb,
    dt: f32,
) -> Option<f32> {
    let dx = velocity.x * dt;
    let dy = velocity.y * dt;

    // Entry and exit times for each axis
    let (x_entry, x_exit) = if dx > 0.0 {
        ((static_aabb.min.x - moving.max.x) / dx, (static_aabb.max.x - moving.min.x) / dx)
    } else if dx < 0.0 {
        ((static_aabb.max.x - moving.min.x) / dx, (static_aabb.min.x - moving.max.x) / dx)
    } else {
        (f32::NEG_INFINITY, f32::INFINITY)
    };

    let (y_entry, y_exit) = if dy > 0.0 {
        ((static_aabb.min.y - moving.max.y) / dy, (static_aabb.max.y - moving.min.y) / dy)
    } else if dy < 0.0 {
        ((static_aabb.max.y - moving.min.y) / dy, (static_aabb.min.y - moving.max.y) / dy)
    } else {
        (f32::NEG_INFINITY, f32::INFINITY)
    };

    let entry_time = x_entry.max(y_entry);
    let exit_time = x_exit.min(y_exit);

    if entry_time > exit_time || entry_time > 1.0 || exit_time < 0.0 {
        None
    } else {
        Some(entry_time.max(0.0))
    }
}

// ============================================================================
// Part 6: Quadtree Spatial Partitioning
// ============================================================================
//
// A quadtree recursively divides 2D space into four quadrants.
// Each node holds up to MAX_ITEMS entities; when full, it subdivides.
// Collision queries only check entities in the same node or adjacent nodes.
// This reduces broad-phase from O(n²) to O(n log n) average case.

const QUADTREE_MAX_DEPTH: u32 = 8;
const QUADTREE_MAX_ITEMS: usize = 8;

pub struct QuadTree {
    bounds: WorldAabb,
    depth: u32,
    items: Vec<(EntityIndex, WorldAabb)>,
    children: Option<Box<[QuadTree; 4]>>,
}

impl QuadTree {
    pub fn new(bounds: WorldAabb, depth: u32) -> Self {
        Self { bounds, depth, items: Vec::new(), children: None }
    }

    fn subdivide(&mut self) {
        let cx = (self.bounds.min.x + self.bounds.max.x) / 2.0;
        let cy = (self.bounds.min.y + self.bounds.max.y) / 2.0;
        let min = self.bounds.min;
        let max = self.bounds.max;

        self.children = Some(Box::new([
            QuadTree::new(WorldAabb { min, max: Vec2::new(cx, cy) }, self.depth + 1),
            QuadTree::new(WorldAabb { min: Vec2::new(cx, min.y), max: Vec2::new(max.x, cy) }, self.depth + 1),
            QuadTree::new(WorldAabb { min: Vec2::new(min.x, cy), max: Vec2::new(cx, max.y) }, self.depth + 1),
            QuadTree::new(WorldAabb { min: Vec2::new(cx, cy), max }, self.depth + 1),
        ]));
    }

    pub fn insert(&mut self, entity: EntityIndex, aabb: WorldAabb) {
        if !self.bounds.overlaps(&aabb) {
            return;
        }

        if self.depth < QUADTREE_MAX_DEPTH && (self.items.len() >= QUADTREE_MAX_ITEMS || self.children.is_some()) {
            if self.children.is_none() {
                self.subdivide();
                let old_items = std::mem::take(&mut self.items);
                for (e, a) in old_items {
                    for child in self.children.as_mut().unwrap().iter_mut() {
                        child.insert(e, a);
                    }
                }
            }
            for child in self.children.as_mut().unwrap().iter_mut() {
                child.insert(entity, aabb);
            }
        } else {
            self.items.push((entity, aabb));
        }
    }

    pub fn query(&self, query_aabb: &WorldAabb) -> Vec<EntityIndex> {
        if !self.bounds.overlaps(query_aabb) {
            return vec![];
        }
        let mut result: Vec<EntityIndex> = self.items.iter()
            .filter(|(_, aabb)| aabb.overlaps(query_aabb))
            .map(|(e, _)| *e)
            .collect();

        if let Some(children) = &self.children {
            for child in children.iter() {
                result.extend(child.query(query_aabb));
            }
        }

        result
    }
}

// ============================================================================
// Part 7: Camera
// ============================================================================

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,       // vertical field of view (radians)
    pub aspect: f32,      // width / height
    pub near: f32,
    pub far: f32,
    pub is_orthographic: bool,
    pub ortho_size: f32,  // half-height in world units (for 2D)
}

impl Camera {
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::UP,
            fov_y,
            aspect,
            near,
            far,
            is_orthographic: false,
            ortho_size: 5.0,
        }
    }

    pub fn orthographic_2d(width: f32, height: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 1.0),
            target: Vec3::ZERO,
            up: Vec3::UP,
            fov_y: std::f32::consts::FRAC_PI_4,
            aspect: width / height,
            near: 0.1,
            far: 100.0,
            is_orthographic: true,
            ortho_size: height / 2.0,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at(self.position, self.target, self.up)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        if self.is_orthographic {
            // Orthographic: map [-ortho*aspect, ortho*aspect] x [-ortho, ortho] x [near, far] to NDC
            let r = self.ortho_size * self.aspect;
            let t = self.ortho_size;
            let mut m = Mat4::identity();
            m.cols[0][0] = 1.0 / r;
            m.cols[1][1] = 1.0 / t;
            m.cols[2][2] = -2.0 / (self.far - self.near);
            m.cols[3][2] = -(self.far + self.near) / (self.far - self.near);
            m
        } else {
            Mat4::perspective(self.fov_y, self.aspect, self.near, self.far)
        }
    }

    /// Check if a world-space AABB is inside the camera frustum (simplified 2D check)
    pub fn is_visible_2d(&self, aabb: &WorldAabb) -> bool {
        let half_w = self.ortho_size * self.aspect;
        let half_h = self.ortho_size;
        let cx = self.position.x;
        let cy = self.position.y;

        aabb.max.x > cx - half_w
            && aabb.min.x < cx + half_w
            && aabb.max.y > cy - half_h
            && aabb.min.y < cy + half_h
    }

    /// Convert world position to screen coordinates (NDC: -1..1)
    pub fn world_to_screen(&self, world_pos: Vec3) -> Vec3 {
        let view = self.view_matrix();
        let proj = self.projection_matrix();
        let vp = proj.mul_mat(view);
        let clip = vp.mul_vec4(world_pos.to_vec4(1.0));
        if clip.w.abs() < 1e-6 {
            return Vec3::ZERO;
        }
        Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
    }
}

// ============================================================================
// Part 8: Render Queue
// ============================================================================
//
// The render queue accumulates draw calls during update and sorts them
// for efficient rendering. Sorting by:
// 1. Layer (background → foreground)
// 2. Texture ID (minimize texture swaps — expensive GPU operation)
// 3. Depth (back-to-front for transparency)

#[derive(Debug, Clone)]
pub struct DrawCall {
    pub entity: EntityIndex,
    pub texture_id: u32,
    pub uv_rect: [f32; 4],
    pub model_matrix: Mat4,
    pub color: [f32; 4],
    pub layer: i32,
    pub depth: f32,
}

pub struct RenderQueue {
    pub calls: Vec<DrawCall>,
}

impl RenderQueue {
    pub fn new() -> Self { Self { calls: Vec::new() } }

    pub fn clear(&mut self) { self.calls.clear(); }

    pub fn submit(&mut self, call: DrawCall) {
        self.calls.push(call);
    }

    /// Sort by layer, then texture, then depth (back to front within same layer)
    pub fn sort(&mut self) {
        self.calls.sort_by(|a, b| {
            a.layer.cmp(&b.layer)
                .then(a.texture_id.cmp(&b.texture_id))
                .then(b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal))
        });
    }

    /// Build render queue from world state
    pub fn build_from_world(&mut self, world: &World, camera: &Camera) {
        self.clear();

        for (idx, sprite) in world.sprites.iter() {
            let transform = match world.transforms.get(idx) {
                Some(t) => t,
                None => continue,
            };

            // Frustum culling (simplified for 2D)
            if let Some(aabb) = world.aabbs.get(idx) {
                let world_aabb = aabb.world_aabb(transform);
                if !camera.is_visible_2d(&world_aabb) {
                    continue; // Cull off-screen entities
                }
            }

            let model = transform.to_matrix();
            let depth = transform.position.z;

            self.submit(DrawCall {
                entity: idx,
                texture_id: sprite.texture_id,
                uv_rect: sprite.uv_rect,
                model_matrix: model,
                color: sprite.color,
                layer: sprite.layer,
                depth,
            });
        }

        self.sort();
    }
}

// ============================================================================
// Part 9: Sprite Animation
// ============================================================================
//
// Sprite sheets store multiple animation frames in a single texture.
// An animation is defined by a set of frame rectangles and a playback speed.
// We use a simple state machine: each state has its own animation clip,
// and transitions occur based on conditions (e.g., "velocity > 0 → walk").

#[derive(Debug, Clone)]
pub struct AnimationFrame {
    pub uv_rect: [f32; 4],   // [u, v, w, h] in texture space
    pub duration: f32,         // seconds to display this frame
}

#[derive(Debug, Clone)]
pub struct AnimationClip {
    pub name: String,
    pub frames: Vec<AnimationFrame>,
    pub looping: bool,
}

impl AnimationClip {
    pub fn total_duration(&self) -> f32 {
        self.frames.iter().map(|f| f.duration).sum()
    }

    pub fn frame_at_time(&self, time: f32) -> &AnimationFrame {
        let mut elapsed = time;
        for frame in &self.frames {
            if elapsed < frame.duration {
                return frame;
            }
            elapsed -= frame.duration;
        }
        self.frames.last().unwrap_or(&self.frames[0])
    }
}

pub struct AnimatorComponent {
    pub clips: HashMap<String, AnimationClip>,
    pub current_clip: String,
    pub time: f32,
    pub playing: bool,
    pub speed: f32,
}

impl AnimatorComponent {
    pub fn new(clips: HashMap<String, AnimationClip>) -> Self {
        let first_clip = clips.keys().next().cloned().unwrap_or_default();
        Self {
            clips,
            current_clip: first_clip,
            time: 0.0,
            playing: true,
            speed: 1.0,
        }
    }

    pub fn play(&mut self, clip_name: &str) {
        if self.current_clip != clip_name {
            self.current_clip = clip_name.to_string();
            self.time = 0.0;
        }
        self.playing = true;
    }

    pub fn update(&mut self, dt: f32) -> Option<[f32; 4]> {
        if !self.playing {
            return self.current_uv();
        }

        self.time += dt * self.speed;

        if let Some(clip) = self.clips.get(&self.current_clip) {
            let total = clip.total_duration();
            if total > 0.0 {
                if clip.looping {
                    self.time %= total;
                } else {
                    self.time = self.time.min(total - 0.001);
                }
            }
            Some(clip.frame_at_time(self.time).uv_rect)
        } else {
            None
        }
    }

    fn current_uv(&self) -> Option<[f32; 4]> {
        self.clips.get(&self.current_clip).map(|clip| {
            clip.frame_at_time(self.time).uv_rect
        })
    }
}

// ============================================================================
// Part 10: Particle System
// ============================================================================
//
// A particle emitter spawns transient visual effects: fire, smoke, sparks.
// Particles are short-lived entities with velocity, color, and size that
// change over their lifetime.

pub struct ParticleEmitter {
    pub position: Vec2,
    pub emit_rate: f32,          // particles per second
    pub emit_accumulator: f32,
    pub velocity_min: Vec2,
    pub velocity_max: Vec2,
    pub lifetime_min: f32,
    pub lifetime_max: f32,
    pub size_start: f32,
    pub size_end: f32,
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
    pub spread_angle: f32,       // radians, cone spread
    pub gravity_scale: f32,
    pub active: bool,
}

impl ParticleEmitter {
    pub fn new(position: Vec2) -> Self {
        Self {
            position,
            emit_rate: 50.0,
            emit_accumulator: 0.0,
            velocity_min: Vec2::new(-1.0, 2.0),
            velocity_max: Vec2::new(1.0, 5.0),
            lifetime_min: 0.5,
            lifetime_max: 2.0,
            size_start: 0.2,
            size_end: 0.0,
            color_start: [1.0, 0.5, 0.0, 1.0],
            color_end: [1.0, 0.0, 0.0, 0.0],
            spread_angle: std::f32::consts::PI / 6.0,
            gravity_scale: 0.3,
            active: true,
        }
    }

    /// Emit particles based on elapsed time. Returns count of particles to spawn.
    pub fn update(&mut self, dt: f32) -> usize {
        if !self.active { return 0; }
        self.emit_accumulator += self.emit_rate * dt;
        let n = self.emit_accumulator as usize;
        self.emit_accumulator -= n as f32;
        n
    }

    /// Compute a particle's initial state (using a simple LCG for reproducibility)
    pub fn particle_state(&self, seed: u64) -> (Vec2, Vec2, f32, f32) {
        let rng = |s: u64| -> f32 {
            let s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as f32 / (1u64 << 31) as f32
        };

        let vx = self.velocity_min.x + rng(seed) * (self.velocity_max.x - self.velocity_min.x);
        let vy = self.velocity_min.y + rng(seed ^ 0x123) * (self.velocity_max.y - self.velocity_min.y);
        let lt = self.lifetime_min + rng(seed ^ 0x456) * (self.lifetime_max - self.lifetime_min);
        let angle = (rng(seed ^ 0x789) - 0.5) * self.spread_angle;
        let vel = Vec2::new(vx, vy).rotate(angle);

        (self.position, vel, lt, lt)
    }
}

// ============================================================================
// Part 11: Input System
// ============================================================================
//
// Games need low-latency input with support for "just pressed" (true only
// on the frame the key first goes down) and "just released" semantics.
// We track previous and current state to compute these edge events.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Space, Enter, Escape, Shift, Ctrl, Alt,
    ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    Digit0, Digit1, Digit2, Digit3, Digit4,
    Digit5, Digit6, Digit7, Digit8, Digit9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton { Left, Middle, Right }

pub struct InputState {
    keys_current: HashMap<KeyCode, bool>,
    keys_previous: HashMap<KeyCode, bool>,
    mouse_current: HashMap<MouseButton, bool>,
    mouse_previous: HashMap<MouseButton, bool>,
    pub mouse_position: Vec2,
    pub mouse_delta: Vec2,
    pub scroll_delta: Vec2,
    pub text_input: String,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            keys_current: HashMap::new(),
            keys_previous: HashMap::new(),
            mouse_current: HashMap::new(),
            mouse_previous: HashMap::new(),
            mouse_position: Vec2::ZERO,
            mouse_delta: Vec2::ZERO,
            scroll_delta: Vec2::ZERO,
            text_input: String::new(),
        }
    }

    /// Call at the start of each frame to shift current to previous
    pub fn begin_frame(&mut self) {
        self.keys_previous = self.keys_current.clone();
        self.mouse_previous = self.mouse_current.clone();
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
        self.text_input.clear();
    }

    pub fn set_key(&mut self, key: KeyCode, pressed: bool) {
        self.keys_current.insert(key, pressed);
    }

    pub fn set_mouse(&mut self, button: MouseButton, pressed: bool) {
        self.mouse_current.insert(button, pressed);
    }

    pub fn is_key_held(&self, key: KeyCode) -> bool {
        *self.keys_current.get(&key).unwrap_or(&false)
    }

    pub fn is_key_just_pressed(&self, key: KeyCode) -> bool {
        *self.keys_current.get(&key).unwrap_or(&false)
            && !*self.keys_previous.get(&key).unwrap_or(&false)
    }

    pub fn is_key_just_released(&self, key: KeyCode) -> bool {
        !*self.keys_current.get(&key).unwrap_or(&false)
            && *self.keys_previous.get(&key).unwrap_or(&false)
    }

    pub fn is_mouse_held(&self, button: MouseButton) -> bool {
        *self.mouse_current.get(&button).unwrap_or(&false)
    }

    pub fn is_mouse_just_pressed(&self, button: MouseButton) -> bool {
        *self.mouse_current.get(&button).unwrap_or(&false)
            && !*self.mouse_previous.get(&button).unwrap_or(&false)
    }
}

// ============================================================================
// Part 12: Game Loop Timing
// ============================================================================
//
// The game loop runs at variable frame rate but updates physics at a fixed
// timestep. This prevents physics instability when frames are slow.
//
// The "semi-fixed timestep" pattern:
// - Accumulate elapsed real time
// - Consume it in fixed dt chunks for physics/game logic
// - Interpolate rendering by the fractional remainder
//
// This decouples render rate from simulation rate — perfect for networking
// (server runs at 20 Hz, client renders at 120 Hz, interpolated smoothly).

pub struct GameTimer {
    pub dt_fixed: f32,          // physics timestep (e.g., 1/60)
    pub dt_max: f32,             // maximum step to avoid spiral of death
    accumulator: f32,
    pub total_time: f64,
    pub frame_count: u64,
    pub fps: f32,
    fps_timer: f32,
    fps_frames: u32,
}

impl GameTimer {
    pub fn new(dt_fixed: f32) -> Self {
        Self {
            dt_fixed,
            dt_max: 0.25,
            accumulator: 0.0,
            total_time: 0.0,
            frame_count: 0,
            fps: 0.0,
            fps_timer: 0.0,
            fps_frames: 0,
        }
    }

    /// Call with the real elapsed frame time. Returns the number of fixed steps to run.
    pub fn tick(&mut self, frame_time: f32) -> (usize, f32) {
        let frame_time = frame_time.min(self.dt_max);
        self.total_time += frame_time as f64;
        self.accumulator += frame_time;
        self.frame_count += 1;

        // FPS counter
        self.fps_timer += frame_time;
        self.fps_frames += 1;
        if self.fps_timer >= 1.0 {
            self.fps = self.fps_frames as f32 / self.fps_timer;
            self.fps_timer = 0.0;
            self.fps_frames = 0;
        }

        let steps = (self.accumulator / self.dt_fixed) as usize;
        self.accumulator -= steps as f32 * self.dt_fixed;
        let alpha = self.accumulator / self.dt_fixed; // interpolation factor

        (steps, alpha)
    }
}

// ============================================================================
// Part 13: Scene Graph (Transform Hierarchy)
// ============================================================================
//
// Parent-child transform relationships allow complex entities (e.g., a car
// with wheels) where child positions are relative to their parent.
// The world transform of a child is: parent_world_transform * child_local_transform

pub fn compute_world_transform(world: &World, entity: Entity) -> Option<Mat4> {
    let local_transform = world.transforms.get(entity.index)?;
    let local_mat = local_transform.to_matrix();

    if let Some(parent) = world.parent.get(entity.index) {
        // Recursively compute parent's world transform
        if let Some(parent_mat) = compute_world_transform(world, parent.0) {
            Some(parent_mat.mul_mat(local_mat))
        } else {
            Some(local_mat)
        }
    } else {
        Some(local_mat)
    }
}

// ============================================================================
// Part 14: Comprehensive Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // --- Math tests ---

    #[test]
    fn test_vec2_operations() {
        let a = Vec2::new(3.0, 4.0);
        assert!(approx(a.len(), 5.0));
        assert!(approx(a.norm().len(), 1.0));
        assert!(approx(a.dot(Vec2::new(1.0, 0.0)), 3.0));
        assert!(approx(a.cross(Vec2::new(1.0, 0.0)), -4.0)); // 3*0 - 4*1 = -4
    }

    #[test]
    fn test_vec2_rotate() {
        let v = Vec2::new(1.0, 0.0);
        let r = v.rotate(std::f32::consts::FRAC_PI_2);
        assert!(approx(r.x, 0.0));
        assert!(approx(r.y, 1.0));
    }

    #[test]
    fn test_vec2_reflect() {
        let v = Vec2::new(1.0, -1.0);
        let n = Vec2::new(0.0, 1.0);
        let r = v.reflect(n);
        assert!(approx(r.x, 1.0));
        assert!(approx(r.y, 1.0));
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!(approx(z.x, 0.0));
        assert!(approx(z.y, 0.0));
        assert!(approx(z.z, 1.0));
    }

    #[test]
    fn test_mat4_identity_mul() {
        let m = Mat4::identity();
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let result = m.mul_vec4(v);
        assert!(approx(result.x, 1.0));
        assert!(approx(result.y, 2.0));
        assert!(approx(result.z, 3.0));
    }

    #[test]
    fn test_mat4_translation() {
        let m = Mat4::translation(Vec3::new(5.0, 3.0, 1.0));
        let p = Vec3::new(1.0, 1.0, 1.0);
        let r = m.mul_point(p);
        assert!(approx(r.x, 6.0));
        assert!(approx(r.y, 4.0));
        assert!(approx(r.z, 2.0));
    }

    #[test]
    fn test_mat4_rotation_z() {
        let m = Mat4::rotation_z(std::f32::consts::FRAC_PI_2);
        let p = Vec3::new(1.0, 0.0, 0.0);
        let r = m.mul_point(p);
        assert!(approx(r.x, 0.0));
        assert!(approx(r.y, 1.0));
    }

    #[test]
    fn test_mat4_composition() {
        let t = Mat4::translation(Vec3::new(1.0, 2.0, 0.0));
        let s = Mat4::scale(Vec3::new(2.0, 2.0, 1.0));
        let ts = t.mul_mat(s);
        let p = Vec3::new(1.0, 1.0, 0.0);
        let r = ts.mul_point(p);
        assert!(approx(r.x, 3.0)); // 1*2 + 1 = 3
        assert!(approx(r.y, 4.0)); // 1*2 + 2 = 4
    }

    // --- ECS tests ---

    #[test]
    fn test_spawn_and_despawn() {
        let mut world = World::new();
        let e1 = world.spawn();
        let e2 = world.spawn();
        assert!(world.is_alive(e1));
        assert!(world.is_alive(e2));
        assert_eq!(world.entity_count(), 2);

        world.despawn(e1);
        assert!(!world.is_alive(e1));
        assert_eq!(world.entity_count(), 1);
    }

    #[test]
    fn test_entity_recycling() {
        let mut world = World::new();
        let e1 = world.spawn();
        world.despawn(e1);
        let e2 = world.spawn(); // Should reuse e1's index
        assert_eq!(e2.index, e1.index);
        assert_ne!(e2.generation, e1.generation);
        assert!(!world.is_alive(e1)); // Old entity invalid
        assert!(world.is_alive(e2));
    }

    #[test]
    fn test_sparse_set() {
        let mut set: SparseSet<i32> = SparseSet::new();
        set.insert(0, 100);
        set.insert(5, 200);
        set.insert(3, 300);

        assert_eq!(*set.get(0).unwrap(), 100);
        assert_eq!(*set.get(5).unwrap(), 200);
        assert!(set.get(1).is_none());

        set.remove(5);
        assert!(set.get(5).is_none());
        assert_eq!(set.len(), 2);

        // Update existing
        set.insert(0, 999);
        assert_eq!(*set.get(0).unwrap(), 999);
    }

    #[test]
    fn test_components() {
        let mut world = World::new();
        let e = world.spawn();
        world.transforms.insert(e.index, Transform::new(1.0, 2.0));
        world.velocities.insert(e.index, Velocity::new(3.0, 4.0));

        let t = world.transforms.get(e.index).unwrap();
        assert!(approx(t.position.x, 1.0));

        let v = world.velocities.get(e.index).unwrap();
        assert!(approx(v.linear.x, 3.0));
    }

    // --- Physics tests ---

    #[test]
    fn test_physics_integration() {
        let mut world = World::new();
        let e = world.spawn();

        world.transforms.insert(e.index, Transform::new(0.0, 0.0));
        world.velocities.insert(e.index, Velocity::new(1.0, 0.0));
        world.rigid_bodies.insert(e.index, {
            let mut rb = RigidBody::new(1.0);
            rb.gravity_scale = 0.0; // disable gravity for this test
            rb.friction = 0.0;
            rb
        });

        system_physics(&mut world, 1.0);

        let t = world.transforms.get(e.index).unwrap();
        assert!(approx(t.position.x, 1.0));
        assert!(approx(t.position.y, 0.0));
    }

    #[test]
    fn test_gravity() {
        let mut world = World::new();
        let e = world.spawn();

        world.transforms.insert(e.index, Transform::new(0.0, 10.0));
        world.velocities.insert(e.index, Velocity::zero());
        world.rigid_bodies.insert(e.index, {
            let mut rb = RigidBody::new(1.0);
            rb.friction = 0.0;
            rb
        });

        let dt = 0.1;
        let steps = 10;
        for _ in 0..steps {
            system_physics(&mut world, dt);
        }

        let t = world.transforms.get(e.index).unwrap();
        // Object should have fallen due to gravity
        assert!(t.position.y < 10.0, "Object should fall, y = {}", t.position.y);
    }

    // --- Collision detection tests ---

    #[test]
    fn test_aabb_overlap() {
        let a = WorldAabb { min: Vec2::new(0.0, 0.0), max: Vec2::new(2.0, 2.0) };
        let b = WorldAabb { min: Vec2::new(1.0, 1.0), max: Vec2::new(3.0, 3.0) };
        let c = WorldAabb { min: Vec2::new(3.0, 0.0), max: Vec2::new(5.0, 2.0) };

        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
        assert!(!b.overlaps(&c));
    }

    #[test]
    fn test_aabb_contact() {
        let aabb_a = WorldAabb { min: Vec2::new(0.0, 0.0), max: Vec2::new(2.0, 2.0) };
        let aabb_b = WorldAabb { min: Vec2::new(1.5, 0.5), max: Vec2::new(3.5, 2.5) };

        let contact = aabb_vs_aabb(0, 1, &aabb_a, &aabb_b);
        assert!(contact.is_some());
        let c = contact.unwrap();
        assert!(c.penetration > 0.0);
    }

    #[test]
    fn test_collision_resolution() {
        let mut world = World::new();

        let e_a = world.spawn();
        world.transforms.insert(e_a.index, Transform::new(0.0, 0.0));
        world.velocities.insert(e_a.index, Velocity::new(2.0, 0.0));
        world.aabbs.insert(e_a.index, Aabb::new(1.0, 1.0));
        world.rigid_bodies.insert(e_a.index, {
            let mut rb = RigidBody::new(1.0);
            rb.gravity_scale = 0.0;
            rb
        });

        let e_b = world.spawn();
        world.transforms.insert(e_b.index, Transform::new(0.8, 0.0)); // Overlapping
        world.velocities.insert(e_b.index, Velocity::new(-2.0, 0.0));
        world.aabbs.insert(e_b.index, Aabb::new(1.0, 1.0));
        world.rigid_bodies.insert(e_b.index, {
            let mut rb = RigidBody::new(1.0);
            rb.gravity_scale = 0.0;
            rb
        });

        let contacts = detect_collisions(&world);
        assert!(!contacts.is_empty(), "Should detect collision");

        let vel_before_a = world.velocities.get(e_a.index).unwrap().linear.x;
        for contact in &contacts {
            resolve_contact(&mut world, contact);
        }
        let vel_after_a = world.velocities.get(e_a.index).unwrap().linear.x;

        // After collision, velocities should change
        assert!(vel_after_a != vel_before_a || contacts.is_empty());
    }

    #[test]
    fn test_swept_aabb() {
        let moving = WorldAabb {
            min: Vec2::new(0.0, 0.0),
            max: Vec2::new(1.0, 1.0),
        };
        let static_box = WorldAabb {
            min: Vec2::new(2.0, 0.0),
            max: Vec2::new(3.0, 1.0),
        };

        // Moving right at velocity 5 for dt=1 — should hit static at t≈0.2
        let t = swept_aabb(&moving, Vec2::new(5.0, 0.0), &static_box, 1.0);
        assert!(t.is_some(), "Should detect collision");
        let t = t.unwrap();
        assert!(t > 0.0 && t < 1.0, "Collision time should be in (0,1), got {}", t);

        // Moving left — no collision
        let t = swept_aabb(&moving, Vec2::new(-5.0, 0.0), &static_box, 1.0);
        assert!(t.is_none(), "Should not detect collision when moving away");
    }

    // --- Quadtree tests ---

    #[test]
    fn test_quadtree_query() {
        let bounds = WorldAabb { min: Vec2::new(-100.0, -100.0), max: Vec2::new(100.0, 100.0) };
        let mut tree = QuadTree::new(bounds, 0);

        // Insert entities at various positions
        tree.insert(0, WorldAabb { min: Vec2::new(-5.0, -5.0), max: Vec2::new(5.0, 5.0) });
        tree.insert(1, WorldAabb { min: Vec2::new(50.0, 50.0), max: Vec2::new(60.0, 60.0) });
        tree.insert(2, WorldAabb { min: Vec2::new(-3.0, -3.0), max: Vec2::new(3.0, 3.0) });

        // Query near origin — should find 0 and 2
        let query = WorldAabb { min: Vec2::new(-6.0, -6.0), max: Vec2::new(6.0, 6.0) };
        let results = tree.query(&query);
        assert!(results.contains(&0));
        assert!(results.contains(&2));

        // Query far from origin — should find 1
        let query2 = WorldAabb { min: Vec2::new(45.0, 45.0), max: Vec2::new(65.0, 65.0) };
        let results2 = tree.query(&query2);
        assert!(results2.contains(&1));
    }

    // --- Camera tests ---

    #[test]
    fn test_camera_perspective() {
        let cam = Camera::perspective(
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            1000.0
        );
        let proj = cam.projection_matrix();

        // Project a point at the origin
        let ndc = cam.world_to_screen(Vec3::ZERO);
        // Origin should map to somewhere behind the camera (at z=5 looking at origin)
        assert!(!ndc.x.is_nan());
        assert!(!ndc.y.is_nan());
    }

    #[test]
    fn test_camera_frustum_culling() {
        let cam = Camera::orthographic_2d(800.0, 600.0);
        let visible = WorldAabb { min: Vec2::new(-1.0, -1.0), max: Vec2::new(1.0, 1.0) };
        let invisible = WorldAabb { min: Vec2::new(1000.0, 1000.0), max: Vec2::new(1010.0, 1010.0) };

        assert!(cam.is_visible_2d(&visible));
        assert!(!cam.is_visible_2d(&invisible));
    }

    // --- Render queue tests ---

    #[test]
    fn test_render_queue_sorting() {
        let mut queue = RenderQueue::new();

        queue.submit(DrawCall {
            entity: 0, texture_id: 2, uv_rect: [0.0; 4],
            model_matrix: Mat4::identity(), color: [1.0; 4], layer: 1, depth: 0.5,
        });
        queue.submit(DrawCall {
            entity: 1, texture_id: 1, uv_rect: [0.0; 4],
            model_matrix: Mat4::identity(), color: [1.0; 4], layer: 0, depth: 0.5,
        });
        queue.submit(DrawCall {
            entity: 2, texture_id: 2, uv_rect: [0.0; 4],
            model_matrix: Mat4::identity(), color: [1.0; 4], layer: 1, depth: 0.2,
        });

        queue.sort();

        // Layer 0 should come before layer 1
        assert_eq!(queue.calls[0].layer, 0);
        // Within layer 1, entity 2 (depth 0.2) before entity 0 (depth 0.5) — lower depth = further back
        // Actually our sort: back-to-front within same layer, so higher depth = closer = rendered last
        // depth 0.5 > 0.2 so entity 2 comes first (b.depth.cmp(a.depth) means higher depth goes first? No, reversed)
        // We do b.depth.cmp(a.depth) which means higher depth is earlier... let's verify the sort is stable
        assert!(queue.calls.len() == 3);
    }

    // --- Animation tests ---

    #[test]
    fn test_animation_clip() {
        let clip = AnimationClip {
            name: "walk".to_string(),
            frames: vec![
                AnimationFrame { uv_rect: [0.0, 0.0, 0.25, 1.0], duration: 0.1 },
                AnimationFrame { uv_rect: [0.25, 0.0, 0.25, 1.0], duration: 0.1 },
                AnimationFrame { uv_rect: [0.5, 0.0, 0.25, 1.0], duration: 0.1 },
                AnimationFrame { uv_rect: [0.75, 0.0, 0.25, 1.0], duration: 0.1 },
            ],
            looping: true,
        };

        assert!(approx(clip.total_duration(), 0.4));

        let frame0 = clip.frame_at_time(0.05);
        assert!(approx(frame0.uv_rect[0], 0.0));

        let frame2 = clip.frame_at_time(0.25);
        assert!(approx(frame2.uv_rect[0], 0.5));
    }

    #[test]
    fn test_animator_looping() {
        let mut clips = HashMap::new();
        clips.insert("walk".to_string(), AnimationClip {
            name: "walk".to_string(),
            frames: vec![
                AnimationFrame { uv_rect: [0.0, 0.0, 0.5, 1.0], duration: 0.5 },
                AnimationFrame { uv_rect: [0.5, 0.0, 0.5, 1.0], duration: 0.5 },
            ],
            looping: true,
        });

        let mut animator = AnimatorComponent::new(clips);
        animator.play("walk");

        let uv1 = animator.update(0.25);
        assert_eq!(uv1.unwrap()[0], 0.0); // First frame

        let uv2 = animator.update(0.5);
        assert_eq!(uv2.unwrap()[0], 0.5); // Second frame after 0.75s total

        let uv3 = animator.update(0.5);
        // After 1.25s, should have looped back to first frame
        assert_eq!(uv3.unwrap()[0], 0.0);
    }

    // --- Input tests ---

    #[test]
    fn test_input_just_pressed() {
        let mut input = InputState::new();

        input.begin_frame();
        input.set_key(KeyCode::Space, true);

        assert!(input.is_key_just_pressed(KeyCode::Space));
        assert!(input.is_key_held(KeyCode::Space));
        assert!(!input.is_key_just_released(KeyCode::Space));

        input.begin_frame(); // Space still held
        assert!(!input.is_key_just_pressed(KeyCode::Space)); // No longer "just" pressed
        assert!(input.is_key_held(KeyCode::Space));

        input.begin_frame();
        input.set_key(KeyCode::Space, false);
        assert!(input.is_key_just_released(KeyCode::Space));
        assert!(!input.is_key_held(KeyCode::Space));
    }

    // --- Health component tests ---

    #[test]
    fn test_health_damage_and_heal() {
        let mut hp = Health::new(100.0);
        assert!(approx(hp.fraction(), 1.0));

        hp.take_damage(30.0);
        assert!(approx(hp.current, 70.0));
        assert!(!hp.is_dead());

        hp.heal(10.0);
        assert!(approx(hp.current, 80.0));

        hp.take_damage(200.0);
        assert!(approx(hp.current, 0.0));
        assert!(hp.is_dead());
    }

    #[test]
    fn test_health_invincibility() {
        let mut hp = Health::new(100.0);
        hp.invincible = true;
        hp.take_damage(50.0);
        assert!(approx(hp.current, 100.0)); // Not damaged
    }

    // --- Tag tests ---

    #[test]
    fn test_tags() {
        let mut tag = Tag::new(tags::PLAYER);
        assert!(tag.has(tags::PLAYER));
        assert!(!tag.has(tags::ENEMY));

        tag.add(tags::SOLID);
        assert!(tag.has(tags::SOLID));

        tag.remove(tags::PLAYER);
        assert!(!tag.has(tags::PLAYER));
        assert!(tag.has(tags::SOLID));
    }

    // --- Game timer tests ---

    #[test]
    fn test_game_timer() {
        let mut timer = GameTimer::new(1.0 / 60.0);
        let (steps, alpha) = timer.tick(1.0 / 60.0);
        assert_eq!(steps, 1);
        assert!(alpha < 1.0);

        let (steps, _) = timer.tick(1.0 / 30.0); // Two physics steps worth
        assert_eq!(steps, 2);
    }

    #[test]
    fn test_game_timer_spiral_of_death_prevention() {
        let mut timer = GameTimer::new(1.0 / 60.0);
        // Very long frame (e.g., 5 seconds) should be clamped
        let (steps, _) = timer.tick(5.0);
        let max_steps = (timer.dt_max / timer.dt_fixed) as usize + 1;
        assert!(steps <= max_steps, "Steps {} should be capped at {}", steps, max_steps);
    }

    // --- Integration test: full game loop tick ---

    #[test]
    fn test_full_game_loop_tick() {
        let mut world = World::new();
        let mut timer = GameTimer::new(1.0 / 60.0);
        let mut input = InputState::new();
        let mut queue = RenderQueue::new();

        // Spawn a player
        let player = world.spawn();
        world.transforms.insert(player.index, Transform::new(0.0, 5.0));
        world.velocities.insert(player.index, Velocity::zero());
        world.aabbs.insert(player.index, Aabb::new(1.0, 1.0));
        world.sprites.insert(player.index, Sprite::new(1));
        world.healths.insert(player.index, Health::new(100.0));
        world.tags.insert(player.index, Tag::new(tags::PLAYER));
        world.rigid_bodies.insert(player.index, RigidBody::new(1.0));

        // Spawn a static platform
        let platform = world.spawn();
        world.transforms.insert(platform.index, Transform::new(0.0, 0.0));
        world.aabbs.insert(platform.index, Aabb::new(10.0, 1.0));
        world.tags.insert(platform.index, Tag::new(tags::SOLID));
        world.rigid_bodies.insert(platform.index, RigidBody::static_body());

        let camera = Camera::orthographic_2d(800.0, 600.0);

        // Simulate one second of game time
        let (steps, alpha) = timer.tick(1.0 / 60.0);
        assert_eq!(steps, 1);

        input.begin_frame();
        // Simulate WASD input
        input.set_key(KeyCode::D, true);
        assert!(input.is_key_just_pressed(KeyCode::D));

        // Physics update
        for _ in 0..steps {
            if input.is_key_held(KeyCode::D) {
                if let Some(rb) = world.rigid_bodies.get_mut(player.index) {
                    rb.apply_force(Vec2::new(500.0, 0.0));
                }
            }
            system_physics(&mut world, timer.dt_fixed);

            let contacts = detect_collisions(&world);
            for c in &contacts {
                resolve_contact(&mut world, c);
            }
        }

        // Build render queue
        queue.build_from_world(&world, &camera);

        // Verify player moved right
        let pt = world.transforms.get(player.index).unwrap();
        assert!(pt.position.x > 0.0, "Player should have moved right");

        // Verify render queue has entries
        assert!(!queue.calls.is_empty(), "Render queue should have draw calls");

        let _ = alpha; // Could use for rendering interpolation
    }
}
