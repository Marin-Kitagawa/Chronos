// chronos-accessibility-engine.rs
//
// Chronos Language — Accessibility Engine
//
// Implements a comprehensive accessibility subsystem covering:
//   • ARIA roles, states, and properties (WAI-ARIA 1.2 specification)
//   • Accessibility tree construction and querying
//   • Accessible name computation (accname-1.1 algorithm)
//   • Keyboard navigation: tab order, roving tabindex, arrow-key patterns
//   • Focus management: focus trapping (modals), skip links, focus restoration
//   • Screen reader support: live regions, announcement queues, verbosity levels
//   • Colour contrast checking (WCAG 2.1 AA/AAA criteria)
//   • Touch target sizing (WCAG 2.5.5 — minimum 44×44 CSS pixels)
//   • Reduced-motion detection and animation throttling
//   • Accessibility audit engine (rule-based linter producing WCAG-tagged violations)
//   • Keyboard shortcut registry with conflict detection
//
// Design principles:
//   • Pure Rust, no external crates beyond std.
//   • All WCAG success criteria are tagged with their SC number (e.g. 1.4.3).
//   • The accessibility tree mirrors the browser AXTree model; nodes hold computed
//     role, name, description, states, and properties.
//   • Colour maths uses relative luminance per IEC 61966-2-1 (sRGB).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// § 1  ARIA Roles  (WAI-ARIA 1.2)
// ─────────────────────────────────────────────────────────────────────────────

/// Every discrete ARIA role defined in WAI-ARIA 1.2.
/// Roles determine the semantics exposed to assistive technologies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AriaRole {
    // --- Abstract roles (not used directly by authors) ---
    Command, Composite, Input, Landmark, Range, Roletype, Section,
    SectionHead, Select, Structure, Widget, Window,

    // --- Document structure ---
    Application, Article, Cell, ColumnHeader, Definition, Directory,
    Document, Feed, Figure, Group, Heading, Img, List, ListItem,
    Math, None, Note, Presentation, Row, RowGroup, RowHeader,
    Separator, Table, Term, Tooltip,

    // --- Landmark roles ---
    Banner, Complementary, ContentInfo, Form, Main, Navigation,
    Region, Search,

    // --- Live region roles ---
    Alert, Log, Marquee, Status, Timer,

    // --- Widget roles ---
    Button, CheckBox, ComboBox, Grid, GridCell, Link, ListBox,
    ListBoxOption, Menu, MenuBar, MenuItem, MenuItemCheckBox,
    MenuItemRadio, Option, ProgressBar, Radio, RadioGroup, ScrollBar,
    SearchBox, Slider, SpinButton, Switch, Tab, TabList, TabPanel,
    TextBox, TreeGrid, TreeItem,

    // --- Window roles ---
    AlertDialog, Dialog,

    /// Unknown/custom role string
    Custom(String),
}

impl AriaRole {
    /// Parse a role from its string representation (case-insensitive).
    pub fn from_str(s: &str) -> AriaRole {
        match s.to_lowercase().as_str() {
            "alert"            => AriaRole::Alert,
            "alertdialog"      => AriaRole::AlertDialog,
            "application"      => AriaRole::Application,
            "article"          => AriaRole::Article,
            "banner"           => AriaRole::Banner,
            "button"           => AriaRole::Button,
            "cell"             => AriaRole::Cell,
            "checkbox"         => AriaRole::CheckBox,
            "columnheader"     => AriaRole::ColumnHeader,
            "combobox"         => AriaRole::ComboBox,
            "complementary"    => AriaRole::Complementary,
            "contentinfo"      => AriaRole::ContentInfo,
            "definition"       => AriaRole::Definition,
            "dialog"           => AriaRole::Dialog,
            "directory"        => AriaRole::Directory,
            "document"         => AriaRole::Document,
            "feed"             => AriaRole::Feed,
            "figure"           => AriaRole::Figure,
            "form"             => AriaRole::Form,
            "grid"             => AriaRole::Grid,
            "gridcell"         => AriaRole::GridCell,
            "group"            => AriaRole::Group,
            "heading"          => AriaRole::Heading,
            "img"              => AriaRole::Img,
            "link"             => AriaRole::Link,
            "list"             => AriaRole::List,
            "listbox"          => AriaRole::ListBox,
            "listitem"         => AriaRole::ListItem,
            "log"              => AriaRole::Log,
            "main"             => AriaRole::Main,
            "marquee"          => AriaRole::Marquee,
            "math"             => AriaRole::Math,
            "menu"             => AriaRole::Menu,
            "menubar"          => AriaRole::MenuBar,
            "menuitem"         => AriaRole::MenuItem,
            "menuitemcheckbox" => AriaRole::MenuItemCheckBox,
            "menuitemradio"    => AriaRole::MenuItemRadio,
            "navigation"       => AriaRole::Navigation,
            "none"             => AriaRole::None,
            "note"             => AriaRole::Note,
            "option"           => AriaRole::Option,
            "presentation"     => AriaRole::Presentation,
            "progressbar"      => AriaRole::ProgressBar,
            "radio"            => AriaRole::Radio,
            "radiogroup"       => AriaRole::RadioGroup,
            "region"           => AriaRole::Region,
            "row"              => AriaRole::Row,
            "rowgroup"         => AriaRole::RowGroup,
            "rowheader"        => AriaRole::RowHeader,
            "scrollbar"        => AriaRole::ScrollBar,
            "search"           => AriaRole::Search,
            "searchbox"        => AriaRole::SearchBox,
            "separator"        => AriaRole::Separator,
            "slider"           => AriaRole::Slider,
            "spinbutton"       => AriaRole::SpinButton,
            "status"           => AriaRole::Status,
            "switch"           => AriaRole::Switch,
            "tab"              => AriaRole::Tab,
            "table"            => AriaRole::Table,
            "tablist"          => AriaRole::TabList,
            "tabpanel"         => AriaRole::TabPanel,
            "term"             => AriaRole::Term,
            "textbox"          => AriaRole::TextBox,
            "timer"            => AriaRole::Timer,
            "tooltip"          => AriaRole::Tooltip,
            "treegrid"         => AriaRole::TreeGrid,
            "treeitem"         => AriaRole::TreeItem,
            other              => AriaRole::Custom(other.to_string()),
        }
    }

    /// Returns true if this role is interactive (i.e. should be keyboard-reachable).
    pub fn is_interactive(&self) -> bool {
        matches!(
            self,
            AriaRole::Button | AriaRole::CheckBox | AriaRole::ComboBox
                | AriaRole::Grid | AriaRole::GridCell | AriaRole::Link
                | AriaRole::ListBox | AriaRole::ListBoxOption | AriaRole::Menu
                | AriaRole::MenuBar | AriaRole::MenuItem | AriaRole::MenuItemCheckBox
                | AriaRole::MenuItemRadio | AriaRole::Option | AriaRole::Radio
                | AriaRole::RadioGroup | AriaRole::ScrollBar | AriaRole::SearchBox
                | AriaRole::Slider | AriaRole::SpinButton | AriaRole::Switch
                | AriaRole::Tab | AriaRole::TabList | AriaRole::TextBox
                | AriaRole::TreeGrid | AriaRole::TreeItem | AriaRole::AlertDialog
                | AriaRole::Dialog
        )
    }

    /// Returns true if this role is a landmark.
    pub fn is_landmark(&self) -> bool {
        matches!(
            self,
            AriaRole::Banner | AriaRole::Complementary | AriaRole::ContentInfo
                | AriaRole::Form | AriaRole::Main | AriaRole::Navigation
                | AriaRole::Region | AriaRole::Search
        )
    }

    /// Returns true if this is a live region role.
    pub fn is_live_region(&self) -> bool {
        matches!(
            self,
            AriaRole::Alert | AriaRole::Log | AriaRole::Marquee
                | AriaRole::Status | AriaRole::Timer
        )
    }

    /// Returns the implicit `aria-live` politeness for live region roles.
    pub fn implicit_live(&self) -> Option<LivePoliteness> {
        match self {
            AriaRole::Alert   => Some(LivePoliteness::Assertive),
            AriaRole::Log     => Some(LivePoliteness::Polite),
            AriaRole::Status  => Some(LivePoliteness::Polite),
            AriaRole::Timer   => Some(LivePoliteness::Off),
            AriaRole::Marquee => Some(LivePoliteness::Off),
            _ => None,
        }
    }
}

impl fmt::Display for AriaRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            AriaRole::Alert          => "alert",
            AriaRole::AlertDialog    => "alertdialog",
            AriaRole::Button         => "button",
            AriaRole::CheckBox       => "checkbox",
            AriaRole::ComboBox       => "combobox",
            AriaRole::Dialog         => "dialog",
            AriaRole::Link           => "link",
            AriaRole::Main           => "main",
            AriaRole::Navigation     => "navigation",
            AriaRole::Radio          => "radio",
            AriaRole::Status         => "status",
            AriaRole::TextBox        => "textbox",
            AriaRole::Custom(s)      => s.as_str(),
            _ => "unknown",
        };
        write!(f, "{}", s)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  ARIA States and Properties
// ─────────────────────────────────────────────────────────────────────────────

/// Tristate boolean used by `aria-checked` and `aria-pressed`.
#[derive(Debug, Clone, PartialEq)]
pub enum Tristate {
    True,
    False,
    Mixed,
}

/// `aria-live` politeness settings — controls how screen readers announce updates.
#[derive(Debug, Clone, PartialEq)]
pub enum LivePoliteness {
    /// Do not announce.
    Off,
    /// Announce after current speech finishes.
    Polite,
    /// Interrupt current speech immediately.
    Assertive,
}

/// `aria-autocomplete` values for comboboxes and textboxes.
#[derive(Debug, Clone, PartialEq)]
pub enum AutoComplete {
    None,
    Inline,
    List,
    Both,
}

/// `aria-orientation` for sliders, scrollbars, etc.
#[derive(Debug, Clone, PartialEq)]
pub enum Orientation {
    Horizontal,
    Vertical,
    Undefined,
}

/// `aria-sort` for column/row headers.
#[derive(Debug, Clone, PartialEq)]
pub enum SortDirection {
    None,
    Ascending,
    Descending,
    Other,
}

/// The complete set of ARIA state/property values for a single node.
/// `None` means the attribute is not set.
#[derive(Debug, Clone, Default)]
pub struct AriaProps {
    // States
    pub checked:      Option<Tristate>,
    pub disabled:     Option<bool>,
    pub expanded:     Option<bool>,
    pub grabbed:      Option<bool>,
    pub hidden:       Option<bool>,
    pub invalid:      Option<bool>,
    pub pressed:      Option<Tristate>,
    pub selected:     Option<bool>,
    pub busy:         Option<bool>,

    // Properties
    pub label:        Option<String>,      // aria-label
    pub labelledby:   Option<Vec<String>>, // aria-labelledby (list of IDs)
    pub describedby:  Option<Vec<String>>, // aria-describedby
    pub controls:     Option<Vec<String>>, // aria-controls
    pub owns:         Option<Vec<String>>, // aria-owns
    pub flowto:       Option<Vec<String>>, // aria-flowto
    pub activedesc:   Option<String>,      // aria-activedescendant
    pub autocomplete: Option<AutoComplete>,
    pub haspopup:     Option<String>,
    pub level:        Option<u8>,          // aria-level (1–6 for headings)
    pub live:         Option<LivePoliteness>,
    pub atomic:       Option<bool>,
    pub relevant:     Option<String>,
    pub orientation:  Option<Orientation>,
    pub readonly:     Option<bool>,
    pub required:     Option<bool>,
    pub sort:         Option<SortDirection>,
    pub valuemin:     Option<f64>,
    pub valuemax:     Option<f64>,
    pub valuenow:     Option<f64>,
    pub valuetext:    Option<String>,
    pub placeholder:  Option<String>,
    pub roledescription: Option<String>,
    pub multiline:    Option<bool>,
    pub multiselectable: Option<bool>,
    pub errormessage: Option<String>,
    pub keyshortcuts: Option<String>,
    pub colcount:     Option<i32>,
    pub colindex:     Option<i32>,
    pub colspan:      Option<i32>,
    pub rowcount:     Option<i32>,
    pub rowindex:     Option<i32>,
    pub rowspan:      Option<i32>,
    pub setsize:      Option<i32>,
    pub posinset:     Option<i32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Accessibility Tree
// ─────────────────────────────────────────────────────────────────────────────

/// A unique identifier for nodes in the accessibility tree.
pub type NodeId = usize;

/// The visibility/inclusion state of a node in the accessibility tree.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeVisibility {
    /// Included in the tree and exposed to AT.
    Visible,
    /// Hidden from AT (aria-hidden=true, display:none, visibility:hidden).
    Hidden,
    /// Generic container — passes through to children.
    Presentational,
}

/// Bounding box in CSS pixels (used for touch target checking and layout).
#[derive(Debug, Clone, Default)]
pub struct BoundingBox {
    pub x:      f64,
    pub y:      f64,
    pub width:  f64,
    pub height: f64,
}

impl BoundingBox {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        BoundingBox { x, y, width: w, height: h }
    }

    /// Checks WCAG 2.5.5 — touch targets should be at least 44×44 CSS px.
    pub fn meets_touch_target(&self) -> bool {
        self.width >= 44.0 && self.height >= 44.0
    }
}

/// A single node in the accessibility (AX) tree.
#[derive(Debug, Clone)]
pub struct AXNode {
    pub id:           NodeId,
    pub html_tag:     String,
    pub role:         AriaRole,
    pub props:        AriaProps,
    /// Computed accessible name (accname-1.1).
    pub computed_name: Option<String>,
    /// Computed accessible description.
    pub computed_desc: Option<String>,
    pub visibility:   NodeVisibility,
    pub tab_index:    Option<i32>,
    pub parent:       Option<NodeId>,
    pub children:     Vec<NodeId>,
    pub bounds:       BoundingBox,
    /// Raw text content (for name computation from subtree).
    pub text_content: String,
}

impl AXNode {
    pub fn new(id: NodeId, tag: &str, role: AriaRole) -> Self {
        AXNode {
            id,
            html_tag: tag.to_string(),
            role,
            props: AriaProps::default(),
            computed_name: None,
            computed_desc: None,
            visibility: NodeVisibility::Visible,
            tab_index: None,
            parent: None,
            children: Vec::new(),
            bounds: BoundingBox::default(),
            text_content: String::new(),
        }
    }

    /// Returns true if this node is keyboard focusable (either natively or via tabindex≥0).
    pub fn is_focusable(&self) -> bool {
        if self.visibility != NodeVisibility::Visible { return false; }
        if self.props.disabled == Some(true) { return false; }
        if let Some(ti) = self.tab_index {
            return ti >= 0;
        }
        // Natively focusable elements
        matches!(
            self.html_tag.to_lowercase().as_str(),
            "a" | "button" | "input" | "select" | "textarea" | "details" | "summary"
        )
    }
}

/// The full accessibility tree for a document.
pub struct AXTree {
    nodes:    Vec<AXNode>,
    root:     Option<NodeId>,
    id_map:   HashMap<String, NodeId>, // html id → node id
}

impl AXTree {
    pub fn new() -> Self {
        AXTree {
            nodes:  Vec::new(),
            root:   None,
            id_map: HashMap::new(),
        }
    }

    /// Add a node and return its ID.
    pub fn add_node(&mut self, mut node: AXNode) -> NodeId {
        let id = self.nodes.len();
        node.id = id;
        self.nodes.push(node);
        id
    }

    /// Register an HTML `id` attribute → node mapping.
    pub fn register_id(&mut self, html_id: &str, node_id: NodeId) {
        self.id_map.insert(html_id.to_string(), node_id);
    }

    pub fn set_root(&mut self, id: NodeId) { self.root = Some(id); }

    pub fn get(&self, id: NodeId) -> Option<&AXNode> { self.nodes.get(id) }
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut AXNode> { self.nodes.get_mut(id) }

    pub fn node_by_html_id(&self, html_id: &str) -> Option<&AXNode> {
        self.id_map.get(html_id).and_then(|&id| self.get(id))
    }

    /// Link parent ↔ child.
    pub fn add_child(&mut self, parent_id: NodeId, child_id: NodeId) {
        if let Some(child) = self.get_mut(child_id) {
            child.parent = Some(parent_id);
        }
        if let Some(parent) = self.get_mut(parent_id) {
            parent.children.push(child_id);
        }
    }

    /// Compute accessible names for all nodes using accname-1.1.
    pub fn compute_names(&mut self) {
        let n = self.nodes.len();
        for id in 0..n {
            let name = self.compute_name_for(id);
            let desc = self.compute_desc_for(id);
            self.nodes[id].computed_name = name;
            self.nodes[id].computed_desc = desc;
        }
    }

    /// accname-1.1 §4.3 — Accessible Name Computation for a single node.
    ///
    /// Precedence (highest first):
    ///  1. `aria-labelledby`
    ///  2. `aria-label`
    ///  3. Native labelling (HTML `<label>`, `alt`, `title`, `placeholder`)
    ///  4. Text subtree (for roles that support it)
    fn compute_name_for(&self, id: NodeId) -> Option<String> {
        let node = self.nodes.get(id)?;

        // 1. aria-labelledby: concatenate text content of referenced elements.
        if let Some(ids) = &node.props.labelledby {
            let parts: Vec<String> = ids.iter()
                .filter_map(|html_id| {
                    self.id_map.get(html_id)
                        .and_then(|&nid| self.nodes.get(nid))
                        .map(|n| n.text_content.clone())
                })
                .filter(|s| !s.is_empty())
                .collect();
            if !parts.is_empty() {
                return Some(parts.join(" "));
            }
        }

        // 2. aria-label
        if let Some(label) = &node.props.label {
            if !label.trim().is_empty() {
                return Some(label.trim().to_string());
            }
        }

        // 3. Element-specific native name
        match node.html_tag.to_lowercase().as_str() {
            "img" => {
                // alt attribute stored in text_content for this implementation
                if !node.text_content.is_empty() {
                    return Some(node.text_content.clone());
                }
            }
            "input" => {
                if let Some(ph) = &node.props.placeholder {
                    return Some(ph.clone());
                }
            }
            _ => {}
        }

        // 4. Subtree text content (for roles that support name-from-contents)
        if self.role_supports_name_from_contents(&node.role) && !node.text_content.is_empty() {
            return Some(node.text_content.trim().to_string());
        }

        None
    }

    fn compute_desc_for(&self, id: NodeId) -> Option<String> {
        let node = self.nodes.get(id)?;
        if let Some(ids) = &node.props.describedby {
            let parts: Vec<String> = ids.iter()
                .filter_map(|html_id| {
                    self.id_map.get(html_id)
                        .and_then(|&nid| self.nodes.get(nid))
                        .map(|n| n.text_content.clone())
                })
                .filter(|s| !s.is_empty())
                .collect();
            if !parts.is_empty() {
                return Some(parts.join(" "));
            }
        }
        None
    }

    /// WAI-ARIA 1.2 §6.2.3 — Name From: contents (only certain roles support this).
    fn role_supports_name_from_contents(&self, role: &AriaRole) -> bool {
        matches!(
            role,
            AriaRole::Button | AriaRole::Cell | AriaRole::CheckBox
                | AriaRole::ColumnHeader | AriaRole::GridCell | AriaRole::Heading
                | AriaRole::Link | AriaRole::ListItem | AriaRole::MenuItem
                | AriaRole::MenuItemCheckBox | AriaRole::MenuItemRadio
                | AriaRole::Option | AriaRole::Radio | AriaRole::Row
                | AriaRole::RowHeader | AriaRole::Switch | AriaRole::Tab
                | AriaRole::Term | AriaRole::Tooltip | AriaRole::TreeItem
        )
    }

    /// BFS traversal yielding node IDs in document order.
    pub fn bfs_order(&self) -> Vec<NodeId> {
        let mut result = Vec::new();
        let start = match self.root { Some(r) => r, None => return result };
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(id) = queue.pop_front() {
            result.push(id);
            if let Some(node) = self.get(id) {
                for &child in &node.children {
                    queue.push_back(child);
                }
            }
        }
        result
    }

    /// Collect all focusable nodes in document (tab) order.
    /// Tab order = tabindex>0 sorted numerically, then tabindex=0 in document order.
    pub fn tab_order(&self) -> Vec<NodeId> {
        let document_order = self.bfs_order();
        let mut positive_tabs: Vec<(i32, NodeId)> = Vec::new();
        let mut zero_tabs: Vec<NodeId> = Vec::new();

        for id in &document_order {
            if let Some(node) = self.get(*id) {
                if !node.is_focusable() { continue; }
                match node.tab_index {
                    Some(ti) if ti > 0 => positive_tabs.push((ti, *id)),
                    _ => zero_tabs.push(*id),
                }
            }
        }

        // Sort positive tabindex ascending (stable sort preserves document order for ties)
        positive_tabs.sort_by_key(|&(ti, _)| ti);

        let mut order: Vec<NodeId> = positive_tabs.into_iter().map(|(_, id)| id).collect();
        order.extend(zero_tabs);
        order
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Focus Management
// ─────────────────────────────────────────────────────────────────────────────

/// Focus trap context — restricts keyboard focus to a subset of nodes (e.g. a modal dialog).
pub struct FocusTrap {
    /// Nodes inside the trap.
    pub trapped_nodes: Vec<NodeId>,
    /// Node that had focus before the trap was activated (for restoration).
    pub previous_focus: Option<NodeId>,
    /// Index of the currently focused node within `trapped_nodes`.
    pub current_index: usize,
    pub active: bool,
}

impl FocusTrap {
    pub fn new(nodes: Vec<NodeId>, previous: Option<NodeId>) -> Self {
        FocusTrap {
            trapped_nodes: nodes,
            previous_focus: previous,
            current_index: 0,
            active: true,
        }
    }

    /// Advance focus forward (Tab key). Wraps around.
    pub fn next(&mut self) -> Option<NodeId> {
        if self.trapped_nodes.is_empty() { return None; }
        self.current_index = (self.current_index + 1) % self.trapped_nodes.len();
        Some(self.trapped_nodes[self.current_index])
    }

    /// Move focus backward (Shift+Tab). Wraps around.
    pub fn prev(&mut self) -> Option<NodeId> {
        if self.trapped_nodes.is_empty() { return None; }
        if self.current_index == 0 {
            self.current_index = self.trapped_nodes.len() - 1;
        } else {
            self.current_index -= 1;
        }
        Some(self.trapped_nodes[self.current_index])
    }

    /// Deactivate the trap and return the node to restore focus to.
    pub fn release(&mut self) -> Option<NodeId> {
        self.active = false;
        self.previous_focus
    }

    pub fn current(&self) -> Option<NodeId> {
        self.trapped_nodes.get(self.current_index).copied()
    }
}

/// Manages application-level focus state.
pub struct FocusManager {
    pub focused_node:  Option<NodeId>,
    pub focus_history: Vec<NodeId>,
    pub traps:         Vec<FocusTrap>,
    /// Skip-link targets: (label, target_node_id)
    pub skip_links:    Vec<(String, NodeId)>,
}

impl FocusManager {
    pub fn new() -> Self {
        FocusManager {
            focused_node:  None,
            focus_history: Vec::new(),
            traps:         Vec::new(),
            skip_links:    Vec::new(),
        }
    }

    /// Move focus to a node, recording history.
    pub fn focus(&mut self, id: NodeId) {
        if let Some(prev) = self.focused_node {
            self.focus_history.push(prev);
        }
        self.focused_node = Some(id);
    }

    /// Restore focus to the previously focused node.
    pub fn restore(&mut self) -> Option<NodeId> {
        if let Some(prev) = self.focus_history.pop() {
            self.focused_node = Some(prev);
            return Some(prev);
        }
        None
    }

    /// Push a new focus trap for modal dialogs.
    pub fn push_trap(&mut self, nodes: Vec<NodeId>) {
        let prev = self.focused_node;
        let trap = FocusTrap::new(nodes, prev);
        if let Some(first) = trap.trapped_nodes.first().copied() {
            self.focused_node = Some(first);
        }
        self.traps.push(trap);
    }

    /// Pop the top focus trap (dialog closed).
    pub fn pop_trap(&mut self) {
        if let Some(mut trap) = self.traps.pop() {
            if let Some(prev) = trap.release() {
                self.focused_node = Some(prev);
            }
        }
    }

    /// Returns true if a focus trap is currently active.
    pub fn is_trapped(&self) -> bool {
        self.traps.last().map(|t| t.active).unwrap_or(false)
    }

    /// Tab key press — advances focus within a trap or globally.
    pub fn tab(&mut self, tab_order: &[NodeId], shift: bool) -> Option<NodeId> {
        if let Some(trap) = self.traps.last_mut() {
            if trap.active {
                let next = if shift { trap.prev() } else { trap.next() };
                self.focused_node = next;
                return next;
            }
        }

        // Global tab navigation
        if tab_order.is_empty() { return None; }
        let current_pos = self.focused_node
            .and_then(|id| tab_order.iter().position(|&x| x == id))
            .unwrap_or(0);

        let next_pos = if shift {
            if current_pos == 0 { tab_order.len() - 1 } else { current_pos - 1 }
        } else {
            (current_pos + 1) % tab_order.len()
        };

        let next = tab_order[next_pos];
        self.focus(next);
        Some(next)
    }

    /// Register a skip link (e.g. "Skip to main content").
    pub fn add_skip_link(&mut self, label: &str, target: NodeId) {
        self.skip_links.push((label.to_string(), target));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Roving tabindex
// ─────────────────────────────────────────────────────────────────────────────

/// Implements the roving tabindex pattern used in toolbars, tab lists, menus, etc.
///
/// Only the currently active item has tabindex=0; all others have tabindex=-1.
/// Arrow keys move focus within the composite; Tab/Shift+Tab exit.
pub struct RovingTabindex {
    pub items:   Vec<NodeId>,
    pub current: usize,
    pub wrap:    bool,
}

impl RovingTabindex {
    pub fn new(items: Vec<NodeId>, wrap: bool) -> Self {
        RovingTabindex { items, current: 0, wrap }
    }

    pub fn current_item(&self) -> Option<NodeId> {
        self.items.get(self.current).copied()
    }

    /// Move to next item (ArrowRight / ArrowDown).
    pub fn next(&mut self) -> Option<NodeId> {
        if self.items.is_empty() { return None; }
        if self.current + 1 < self.items.len() {
            self.current += 1;
        } else if self.wrap {
            self.current = 0;
        }
        self.current_item()
    }

    /// Move to previous item (ArrowLeft / ArrowUp).
    pub fn prev(&mut self) -> Option<NodeId> {
        if self.items.is_empty() { return None; }
        if self.current > 0 {
            self.current -= 1;
        } else if self.wrap {
            self.current = self.items.len() - 1;
        }
        self.current_item()
    }

    /// Jump to first item (Home key).
    pub fn first(&mut self) -> Option<NodeId> {
        self.current = 0;
        self.current_item()
    }

    /// Jump to last item (End key).
    pub fn last(&mut self) -> Option<NodeId> {
        if self.items.is_empty() { return None; }
        self.current = self.items.len() - 1;
        self.current_item()
    }

    /// Returns which nodes should have tabindex=0 (current) vs tabindex=-1 (others).
    pub fn tabindex_map(&self) -> HashMap<NodeId, i32> {
        self.items.iter().enumerate().map(|(i, &id)| {
            (id, if i == self.current { 0 } else { -1 })
        }).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  Screen Reader / Live Region Announcements
// ─────────────────────────────────────────────────────────────────────────────

/// Verbosity level for screen reader announcements.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Verbosity {
    Low,     // Only assertive alerts
    Medium,  // Polite + assertive
    High,    // Everything including off (explicit)
}

/// A single announcement to be spoken by the screen reader.
#[derive(Debug, Clone)]
pub struct Announcement {
    pub text:       String,
    pub politeness: LivePoliteness,
    pub atomic:     bool,
}

/// Manages the live region announcement queue.
pub struct AnnouncementQueue {
    /// Polite announcements queued until current speech finishes.
    pub polite:    VecDeque<Announcement>,
    /// Assertive announcements that interrupt current speech.
    pub assertive: VecDeque<Announcement>,
    pub verbosity: Verbosity,
}

impl AnnouncementQueue {
    pub fn new(verbosity: Verbosity) -> Self {
        AnnouncementQueue {
            polite:    VecDeque::new(),
            assertive: VecDeque::new(),
            verbosity,
        }
    }

    /// Queue an announcement according to politeness level.
    pub fn announce(&mut self, text: &str, politeness: LivePoliteness, atomic: bool) {
        match &politeness {
            LivePoliteness::Off => {
                if self.verbosity >= Verbosity::High {
                    self.polite.push_back(Announcement {
                        text: text.to_string(), politeness, atomic,
                    });
                }
            }
            LivePoliteness::Polite => {
                if self.verbosity >= Verbosity::Medium {
                    self.polite.push_back(Announcement {
                        text: text.to_string(), politeness, atomic,
                    });
                }
            }
            LivePoliteness::Assertive => {
                // Assertive always clears the polite queue first
                self.polite.clear();
                self.assertive.push_back(Announcement {
                    text: text.to_string(), politeness, atomic,
                });
            }
        }
    }

    /// Drain the next announcement to be spoken.
    /// Assertive announcements take priority.
    pub fn next(&mut self) -> Option<Announcement> {
        self.assertive.pop_front().or_else(|| self.polite.pop_front())
    }

    pub fn is_empty(&self) -> bool {
        self.assertive.is_empty() && self.polite.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Colour Contrast (WCAG 2.1 SC 1.4.3 / 1.4.6)
// ─────────────────────────────────────────────────────────────────────────────

/// sRGB colour as 0–255 integers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self { Color { r, g, b, a: 255 } }
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self { Color { r, g, b, a } }

    /// Parse a CSS hex colour (#rgb / #rrggbb / #rrggbbaa).
    pub fn from_hex(s: &str) -> Option<Self> {
        let s = s.trim_start_matches('#');
        match s.len() {
            3 => {
                let r = u8::from_str_radix(&s[0..1].repeat(2), 16).ok()?;
                let g = u8::from_str_radix(&s[1..2].repeat(2), 16).ok()?;
                let b = u8::from_str_radix(&s[2..3].repeat(2), 16).ok()?;
                Some(Color::rgb(r, g, b))
            }
            6 => {
                let r = u8::from_str_radix(&s[0..2], 16).ok()?;
                let g = u8::from_str_radix(&s[2..4], 16).ok()?;
                let b = u8::from_str_radix(&s[4..6], 16).ok()?;
                Some(Color::rgb(r, g, b))
            }
            8 => {
                let r = u8::from_str_radix(&s[0..2], 16).ok()?;
                let g = u8::from_str_radix(&s[2..4], 16).ok()?;
                let b = u8::from_str_radix(&s[4..6], 16).ok()?;
                let a = u8::from_str_radix(&s[6..8], 16).ok()?;
                Some(Color::rgba(r, g, b, a))
            }
            _ => None,
        }
    }

    /// Relative luminance per WCAG 2.1 definition (IEC 61966-2-1 sRGB).
    ///
    /// Each channel is linearised: if c/255 ≤ 0.04045 → c/12.92, else ((c+0.055)/1.055)^2.4
    /// L = 0.2126 R + 0.7152 G + 0.0722 B
    pub fn relative_luminance(&self) -> f64 {
        fn linearise(c: u8) -> f64 {
            let v = c as f64 / 255.0;
            if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055f64).powf(2.4) }
        }
        0.2126 * linearise(self.r) + 0.7152 * linearise(self.g) + 0.0722 * linearise(self.b)
    }

    /// Composite alpha: blend this colour over a background (for semi-transparent text).
    pub fn blend_over(&self, bg: Color) -> Color {
        let alpha = self.a as f64 / 255.0;
        let blend = |fg: u8, bg: u8| -> u8 {
            (alpha * fg as f64 + (1.0 - alpha) * bg as f64).round() as u8
        };
        Color::rgb(blend(self.r, bg.r), blend(self.g, bg.g), blend(self.b, bg.b))
    }
}

/// Contrast ratio between two colours.
/// Result ∈ [1.0, 21.0]. Formula: (L1 + 0.05) / (L2 + 0.05) with L1 ≥ L2.
pub fn contrast_ratio(fg: Color, bg: Color) -> f64 {
    let l1 = fg.relative_luminance();
    let l2 = bg.relative_luminance();
    let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
    (lighter + 0.05) / (darker + 0.05)
}

/// WCAG conformance level.
#[derive(Debug, Clone, PartialEq)]
pub enum WcagLevel {
    A,
    AA,
    AAA,
    Fail,
}

/// Text size category — determines the required contrast ratio.
#[derive(Debug, Clone, PartialEq)]
pub enum TextSize {
    /// Normal text: < 18pt (or < 14pt bold).
    Normal,
    /// Large text: ≥ 18pt regular or ≥ 14pt bold.
    Large,
    /// Non-text UI component (icons, borders, focus indicators — SC 1.4.11).
    NonText,
}

/// Evaluate contrast against WCAG 2.1 thresholds.
///
/// | Size     | AA    | AAA   |
/// |----------|-------|-------|
/// | Normal   | 4.5:1 | 7.0:1 |
/// | Large    | 3.0:1 | 4.5:1 |
/// | Non-text | 3.0:1 | —     |
pub fn wcag_level(ratio: f64, size: &TextSize) -> WcagLevel {
    match size {
        TextSize::Normal => {
            if ratio >= 7.0  { WcagLevel::AAA }
            else if ratio >= 4.5 { WcagLevel::AA }
            else { WcagLevel::Fail }
        }
        TextSize::Large => {
            if ratio >= 4.5  { WcagLevel::AAA }
            else if ratio >= 3.0 { WcagLevel::AA }
            else { WcagLevel::Fail }
        }
        TextSize::NonText => {
            if ratio >= 3.0 { WcagLevel::AA }
            else { WcagLevel::Fail }
        }
    }
}

/// Full contrast check result.
#[derive(Debug)]
pub struct ContrastResult {
    pub ratio:    f64,
    pub level:    WcagLevel,
    pub pass_aa:  bool,
    pub pass_aaa: bool,
}

pub fn check_contrast(fg: Color, bg: Color, size: TextSize) -> ContrastResult {
    let ratio = contrast_ratio(fg, bg);
    let level = wcag_level(ratio, &size);
    let pass_aa  = level == WcagLevel::AA || level == WcagLevel::AAA;
    let pass_aaa = level == WcagLevel::AAA;
    ContrastResult { ratio, level, pass_aa, pass_aaa }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Keyboard Shortcut Registry
// ─────────────────────────────────────────────────────────────────────────────

/// Modifier keys bitmask.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Modifiers {
    pub ctrl:  bool,
    pub alt:   bool,
    pub shift: bool,
    pub meta:  bool,
}

impl Modifiers {
    pub const NONE:       Modifiers = Modifiers { ctrl: false, alt: false, shift: false, meta: false };
    pub const CTRL:       Modifiers = Modifiers { ctrl: true,  alt: false, shift: false, meta: false };
    pub const ALT:        Modifiers = Modifiers { ctrl: false, alt: true,  shift: false, meta: false };
    pub const SHIFT:      Modifiers = Modifiers { ctrl: false, alt: false, shift: true,  meta: false };
    pub const CTRL_SHIFT: Modifiers = Modifiers { ctrl: true,  alt: false, shift: true,  meta: false };
}

/// A keyboard shortcut binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyBinding {
    pub key:       String,     // Key name, e.g. "Enter", "Escape", "F1", "k"
    pub modifiers: Modifiers,
}

impl KeyBinding {
    pub fn new(key: &str, modifiers: Modifiers) -> Self {
        KeyBinding { key: key.to_string(), modifiers }
    }
    pub fn simple(key: &str) -> Self { Self::new(key, Modifiers::NONE) }
    pub fn ctrl(key: &str)   -> Self { Self::new(key, Modifiers::CTRL) }
    pub fn alt(key: &str)    -> Self { Self::new(key, Modifiers::ALT) }
}

/// A registered shortcut with its handler description.
#[derive(Debug, Clone)]
pub struct Shortcut {
    pub binding:     KeyBinding,
    pub action:      String,
    pub description: String,
    pub scope:       String,    // e.g. "global", "editor", "dialog"
}

/// Registry for all keyboard shortcuts with conflict detection.
pub struct ShortcutRegistry {
    shortcuts: Vec<Shortcut>,
}

impl ShortcutRegistry {
    pub fn new() -> Self { ShortcutRegistry { shortcuts: Vec::new() } }

    /// Register a shortcut. Returns an error if the binding conflicts within the same scope.
    pub fn register(&mut self, shortcut: Shortcut) -> Result<(), String> {
        for existing in &self.shortcuts {
            if existing.binding == shortcut.binding && existing.scope == shortcut.scope {
                return Err(format!(
                    "Conflict: {:?} already bound to '{}' in scope '{}'",
                    shortcut.binding, existing.action, existing.scope
                ));
            }
        }
        self.shortcuts.push(shortcut);
        Ok(())
    }

    /// Find shortcuts matching a binding (may match multiple scopes).
    pub fn find(&self, binding: &KeyBinding) -> Vec<&Shortcut> {
        self.shortcuts.iter().filter(|s| &s.binding == binding).collect()
    }

    /// List all shortcuts sorted by scope then key.
    pub fn all(&self) -> Vec<&Shortcut> {
        let mut refs: Vec<&Shortcut> = self.shortcuts.iter().collect();
        refs.sort_by(|a, b| a.scope.cmp(&b.scope).then(a.binding.key.cmp(&b.binding.key)));
        refs
    }

    /// Detect all conflicts across all scopes.
    pub fn conflicts(&self) -> Vec<(&Shortcut, &Shortcut)> {
        let mut conflicts = Vec::new();
        for (i, a) in self.shortcuts.iter().enumerate() {
            for b in &self.shortcuts[i+1..] {
                if a.binding == b.binding && a.scope == b.scope {
                    conflicts.push((a, b));
                }
            }
        }
        conflicts
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  Reduced Motion
// ─────────────────────────────────────────────────────────────────────────────

/// Animation preference — mirrors `prefers-reduced-motion` media query.
#[derive(Debug, Clone, PartialEq)]
pub enum MotionPreference {
    NoPreference,
    Reduce,
}

/// Duration limiter respecting reduced-motion preference.
/// WCAG 2.3.3 (AAA): animations can be disabled unless essential.
pub fn animation_duration(preferred_ms: f64, preference: &MotionPreference) -> f64 {
    match preference {
        MotionPreference::NoPreference => preferred_ms,
        MotionPreference::Reduce => {
            // Keep a minimal, imperceptible duration so DOM transitions don't break layout.
            preferred_ms.min(1.0)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 10  Accessibility Audit Engine
// ─────────────────────────────────────────────────────────────────────────────

/// WCAG impact severity of a violation.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Impact {
    Minor,
    Moderate,
    Serious,
    Critical,
}

impl fmt::Display for Impact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self { Impact::Minor => "minor", Impact::Moderate => "moderate",
            Impact::Serious => "serious", Impact::Critical => "critical" };
        write!(f, "{}", s)
    }
}

/// A single accessibility violation found during an audit.
#[derive(Debug, Clone)]
pub struct Violation {
    /// WCAG success criterion, e.g. "1.1.1"
    pub criterion: String,
    pub impact:    Impact,
    pub node_id:   NodeId,
    pub message:   String,
    pub help_url:  String,
}

/// Audit rules — each rule checks a specific WCAG success criterion.
pub struct AccessibilityAuditor;

impl AccessibilityAuditor {
    /// Run all rules against the tree. Returns a list of violations.
    pub fn audit(
        tree: &AXTree,
        fg_colors: &HashMap<NodeId, Color>,
        bg_colors: &HashMap<NodeId, Color>,
    ) -> Vec<Violation> {
        let mut violations = Vec::new();

        let order = tree.bfs_order();
        for id in &order {
            let node = match tree.get(*id) { Some(n) => n, None => continue };

            // SC 1.1.1 — Non-text content must have a text alternative.
            if node.html_tag == "img" {
                if node.computed_name.is_none() || node.computed_name.as_deref() == Some("") {
                    violations.push(Violation {
                        criterion: "1.1.1".into(),
                        impact: Impact::Critical,
                        node_id: *id,
                        message: "Image has no accessible name (missing alt attribute or aria-label)".into(),
                        help_url: "https://www.w3.org/WAI/WCAG21/Understanding/non-text-content".into(),
                    });
                }
            }

            // SC 1.4.3 — Contrast ratio for normal text must be ≥ 4.5:1.
            if let (Some(&fg), Some(&bg)) = (fg_colors.get(id), bg_colors.get(id)) {
                let ratio = contrast_ratio(fg, bg);
                if ratio < 4.5 {
                    let impact = if ratio < 3.0 { Impact::Critical } else { Impact::Serious };
                    violations.push(Violation {
                        criterion: "1.4.3".into(),
                        impact,
                        node_id: *id,
                        message: format!(
                            "Colour contrast ratio is {:.2}:1, requires 4.5:1 (AA) for normal text",
                            ratio
                        ),
                        help_url: "https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum".into(),
                    });
                }
            }

            // SC 2.1.1 — Interactive elements must be keyboard accessible.
            if node.role.is_interactive() && node.props.disabled != Some(true) {
                if node.tab_index == Some(-1) {
                    violations.push(Violation {
                        criterion: "2.1.1".into(),
                        impact: Impact::Critical,
                        node_id: *id,
                        message: format!(
                            "Interactive element with role '{}' has tabindex=-1 and is not keyboard reachable",
                            node.role
                        ),
                        help_url: "https://www.w3.org/WAI/WCAG21/Understanding/keyboard".into(),
                    });
                }
            }

            // SC 2.4.3 — Focus order must be meaningful (detect tabindex > 0 anti-pattern).
            if node.tab_index.unwrap_or(0) > 0 {
                violations.push(Violation {
                    criterion: "2.4.3".into(),
                    impact: Impact::Moderate,
                    node_id: *id,
                    message: format!(
                        "tabindex={} found; positive tabindex values disrupt natural focus order",
                        node.tab_index.unwrap()
                    ),
                    help_url: "https://www.w3.org/WAI/WCAG21/Understanding/focus-order".into(),
                });
            }

            // SC 1.3.1 — Elements using ARIA roles must have required owned elements / context.
            // Example: `listitem` must be inside `list`.
            if node.role == AriaRole::ListItem {
                let parent_ok = node.parent.and_then(|pid| tree.get(pid))
                    .map(|p| p.role == AriaRole::List || p.role == AriaRole::ListBox)
                    .unwrap_or(false);
                if !parent_ok {
                    violations.push(Violation {
                        criterion: "1.3.1".into(),
                        impact: Impact::Serious,
                        node_id: *id,
                        message: "role='listitem' must be contained within role='list'".into(),
                        help_url: "https://www.w3.org/WAI/WCAG21/Understanding/info-and-relationships".into(),
                    });
                }
            }

            // SC 2.5.5 — Touch targets must be at least 44×44 CSS px.
            if node.role.is_interactive() && !node.bounds.meets_touch_target() {
                if node.bounds.width > 0.0 || node.bounds.height > 0.0 {
                    violations.push(Violation {
                        criterion: "2.5.5".into(),
                        impact: Impact::Minor,
                        node_id: *id,
                        message: format!(
                            "Touch target size is {:.0}×{:.0}px, should be at least 44×44px",
                            node.bounds.width, node.bounds.height
                        ),
                        help_url: "https://www.w3.org/WAI/WCAG21/Understanding/target-size".into(),
                    });
                }
            }

            // SC 4.1.2 — Name, Role, Value: all UI components must have an accessible name.
            if node.role.is_interactive() && node.computed_name.is_none() {
                violations.push(Violation {
                    criterion: "4.1.2".into(),
                    impact: Impact::Critical,
                    node_id: *id,
                    message: format!(
                        "Interactive element with role '{}' has no accessible name",
                        node.role
                    ),
                    help_url: "https://www.w3.org/WAI/WCAG21/Understanding/name-role-value".into(),
                });
            }

            // SC 1.4.4 — Hidden interactive elements shouldn't be in tab order.
            if node.props.hidden == Some(true) && node.is_focusable() {
                violations.push(Violation {
                    criterion: "1.4.4".into(),
                    impact: Impact::Serious,
                    node_id: *id,
                    message: "aria-hidden=true element is in the tab order".into(),
                    help_url: "https://www.w3.org/WAI/WCAG21/Understanding/resize-text".into(),
                });
            }
        }

        violations
    }

    /// Summarise violations by criterion.
    pub fn summary(violations: &[Violation]) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for v in violations {
            *map.entry(v.criterion.clone()).or_insert(0) += 1;
        }
        map
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 11  Landmark Navigation
// ─────────────────────────────────────────────────────────────────────────────

/// Provides AT-style landmark navigation (cycling through landmark regions).
pub struct LandmarkNavigator {
    landmarks: Vec<NodeId>,
    current:   usize,
}

impl LandmarkNavigator {
    pub fn from_tree(tree: &AXTree) -> Self {
        let landmarks: Vec<NodeId> = tree.bfs_order()
            .into_iter()
            .filter(|&id| {
                tree.get(id).map(|n| n.role.is_landmark()).unwrap_or(false)
            })
            .collect();
        LandmarkNavigator { landmarks, current: 0 }
    }

    pub fn next_landmark(&mut self) -> Option<NodeId> {
        if self.landmarks.is_empty() { return None; }
        self.current = (self.current + 1) % self.landmarks.len();
        Some(self.landmarks[self.current])
    }

    pub fn prev_landmark(&mut self) -> Option<NodeId> {
        if self.landmarks.is_empty() { return None; }
        if self.current == 0 { self.current = self.landmarks.len() - 1; }
        else { self.current -= 1; }
        Some(self.landmarks[self.current])
    }

    pub fn count(&self) -> usize { self.landmarks.len() }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 12  Heading Navigation
// ─────────────────────────────────────────────────────────────────────────────

/// Heading navigation — similar to screen reader H / 1–6 shortcuts.
pub struct HeadingNavigator {
    /// (level, node_id) sorted in document order.
    headings: Vec<(u8, NodeId)>,
    current:  usize,
}

impl HeadingNavigator {
    pub fn from_tree(tree: &AXTree) -> Self {
        let headings: Vec<(u8, NodeId)> = tree.bfs_order()
            .into_iter()
            .filter_map(|id| {
                let node = tree.get(id)?;
                if node.role == AriaRole::Heading {
                    Some((node.props.level.unwrap_or(1), id))
                } else {
                    None
                }
            })
            .collect();
        HeadingNavigator { headings, current: 0 }
    }

    pub fn next_heading(&mut self, level_filter: Option<u8>) -> Option<NodeId> {
        let start = self.current;
        loop {
            self.current = (self.current + 1) % self.headings.len();
            if self.current == start { return None; }
            let (level, id) = self.headings[self.current];
            if level_filter.map(|f| f == level).unwrap_or(true) {
                return Some(id);
            }
        }
    }

    pub fn count(&self) -> usize { self.headings.len() }

    pub fn outline(&self) -> Vec<(u8, NodeId)> { self.headings.clone() }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 13  Accessible Name Utility Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Strip HTML tags from a string (for extracting plain text name).
pub fn strip_html(s: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => result.push(c),
            _ => {}
        }
    }
    result
}

/// Normalise whitespace: collapse runs of whitespace to a single space, trim ends.
pub fn normalise_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ─────────────────────────────────────────────────────────────────────────────
// § 14  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Colour Contrast ────────────────────────────────────────────────────

    #[test]
    fn test_relative_luminance_white() {
        let white = Color::rgb(255, 255, 255);
        let lum = white.relative_luminance();
        assert!((lum - 1.0).abs() < 1e-6, "white luminance should be 1.0, got {}", lum);
    }

    #[test]
    fn test_relative_luminance_black() {
        let black = Color::rgb(0, 0, 0);
        let lum = black.relative_luminance();
        assert!(lum.abs() < 1e-9, "black luminance should be 0.0, got {}", lum);
    }

    #[test]
    fn test_contrast_ratio_white_on_black() {
        let ratio = contrast_ratio(Color::rgb(255, 255, 255), Color::rgb(0, 0, 0));
        assert!((ratio - 21.0).abs() < 0.01, "white/black ratio should be 21.0, got {:.2}", ratio);
    }

    #[test]
    fn test_contrast_ratio_same_color() {
        let ratio = contrast_ratio(Color::rgb(128, 128, 128), Color::rgb(128, 128, 128));
        assert!((ratio - 1.0).abs() < 0.01, "same colour ratio should be 1.0");
    }

    #[test]
    fn test_wcag_normal_text_aa_pass() {
        // Black on white → 21:1 → AAA
        let ratio = contrast_ratio(Color::rgb(0, 0, 0), Color::rgb(255, 255, 255));
        let result = check_contrast(Color::rgb(0, 0, 0), Color::rgb(255, 255, 255), TextSize::Normal);
        assert!(result.pass_aa);
        assert!(result.pass_aaa);
        assert_eq!(result.level, WcagLevel::AAA);
    }

    #[test]
    fn test_wcag_normal_text_fail() {
        // Light grey on white — very low contrast
        let fg = Color::rgb(200, 200, 200);
        let bg = Color::rgb(255, 255, 255);
        let result = check_contrast(fg, bg, TextSize::Normal);
        assert!(!result.pass_aa);
        assert_eq!(result.level, WcagLevel::Fail);
    }

    #[test]
    fn test_wcag_large_text_lower_threshold() {
        // A colour that passes AA for large text (>=3.0:1) but fails for normal (>=4.5:1).
        // rgb(135,135,135) on white gives approximately 3.7:1, which is in [3.0, 4.5).
        let fg = Color::rgb(135, 135, 135);
        let bg = Color::rgb(255, 255, 255);
        let ratio = contrast_ratio(fg, bg);
        assert!(ratio >= 3.0 && ratio < 4.5, "expected ratio in [3.0, 4.5), got {}", ratio);
        let large  = check_contrast(fg, bg, TextSize::Large);
        let normal = check_contrast(fg, bg, TextSize::Normal);
        assert!(large.pass_aa);
        assert!(!normal.pass_aa);
    }

    #[test]
    fn test_color_from_hex_6() {
        let c = Color::from_hex("#ff0080").unwrap();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 128);
    }

    #[test]
    fn test_color_from_hex_3() {
        let c = Color::from_hex("#f00").unwrap();
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
    }

    #[test]
    fn test_color_blend_over() {
        // 50% transparent red over white → pink
        let fg = Color::rgba(255, 0, 0, 128);
        let bg = Color::rgb(255, 255, 255);
        let blended = fg.blend_over(bg);
        assert!(blended.r > 200);
        assert!(blended.g > 100);
    }

    // ── ARIA Roles ──────────────────────────────────────────────────────────

    #[test]
    fn test_role_from_str_button() {
        assert_eq!(AriaRole::from_str("button"), AriaRole::Button);
    }

    #[test]
    fn test_role_from_str_case_insensitive() {
        assert_eq!(AriaRole::from_str("DIALOG"), AriaRole::Dialog);
        assert_eq!(AriaRole::from_str("Dialog"), AriaRole::Dialog);
    }

    #[test]
    fn test_role_is_interactive() {
        assert!(AriaRole::Button.is_interactive());
        assert!(AriaRole::CheckBox.is_interactive());
        assert!(!AriaRole::Main.is_interactive());
        assert!(!AriaRole::Heading.is_interactive());
    }

    #[test]
    fn test_role_is_landmark() {
        assert!(AriaRole::Main.is_landmark());
        assert!(AriaRole::Navigation.is_landmark());
        assert!(!AriaRole::Button.is_landmark());
    }

    #[test]
    fn test_role_is_live_region() {
        assert!(AriaRole::Alert.is_live_region());
        assert!(AriaRole::Status.is_live_region());
        assert!(!AriaRole::Button.is_live_region());
    }

    #[test]
    fn test_alert_implicit_live_assertive() {
        assert_eq!(AriaRole::Alert.implicit_live(), Some(LivePoliteness::Assertive));
    }

    // ── AXTree construction ─────────────────────────────────────────────────

    fn build_simple_tree() -> AXTree {
        let mut tree = AXTree::new();

        let mut main_node = AXNode::new(0, "main", AriaRole::Main);
        main_node.text_content = "".into();
        let main_id = tree.add_node(main_node);

        let mut btn = AXNode::new(1, "button", AriaRole::Button);
        btn.text_content = "Save".into();
        btn.tab_index = Some(0);
        let btn_id = tree.add_node(btn);

        let mut img = AXNode::new(2, "img", AriaRole::Img);
        img.text_content = "Company logo".into(); // simulates alt=""
        let img_id = tree.add_node(img);

        tree.set_root(main_id);
        tree.add_child(main_id, btn_id);
        tree.add_child(main_id, img_id);
        tree.compute_names();
        tree
    }

    #[test]
    fn test_axtree_button_name_from_text() {
        let tree = build_simple_tree();
        let btn = tree.get(1).unwrap();
        assert_eq!(btn.computed_name.as_deref(), Some("Save"));
    }

    #[test]
    fn test_axtree_img_name_from_alt() {
        let tree = build_simple_tree();
        let img = tree.get(2).unwrap();
        assert_eq!(img.computed_name.as_deref(), Some("Company logo"));
    }

    #[test]
    fn test_axtree_bfs_order() {
        let tree = build_simple_tree();
        let order = tree.bfs_order();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn test_axtree_tab_order_only_focusable() {
        let tree = build_simple_tree();
        let tab = tree.tab_order();
        // Only button has tabindex=0; main and img are not focusable
        assert_eq!(tab, vec![1]);
    }

    #[test]
    fn test_aria_label_overrides_text_content() {
        let mut tree = AXTree::new();
        let mut btn = AXNode::new(0, "button", AriaRole::Button);
        btn.text_content = "X".into();
        btn.props.label = Some("Close dialog".into());
        tree.add_node(btn);
        tree.compute_names();
        assert_eq!(tree.get(0).unwrap().computed_name.as_deref(), Some("Close dialog"));
    }

    #[test]
    fn test_aria_labelledby_highest_priority() {
        let mut tree = AXTree::new();

        let mut label_node = AXNode::new(0, "span", AriaRole::None);
        label_node.text_content = "Full name".into();
        tree.add_node(label_node);
        tree.register_id("lbl", 0);

        let mut input = AXNode::new(1, "input", AriaRole::TextBox);
        input.props.label = Some("should be overridden".into());
        input.props.labelledby = Some(vec!["lbl".into()]);
        tree.add_node(input);
        tree.compute_names();

        assert_eq!(tree.get(1).unwrap().computed_name.as_deref(), Some("Full name"));
    }

    // ── Focus Management ────────────────────────────────────────────────────

    #[test]
    fn test_focus_manager_basic() {
        let mut fm = FocusManager::new();
        fm.focus(1);
        assert_eq!(fm.focused_node, Some(1));
        fm.focus(2);
        assert_eq!(fm.focused_node, Some(2));
        let restored = fm.restore();
        assert_eq!(restored, Some(1));
    }

    #[test]
    fn test_focus_trap_wraps() {
        let mut fm = FocusManager::new();
        fm.push_trap(vec![10, 20, 30]);
        assert_eq!(fm.focused_node, Some(10));

        let trap = fm.traps.last_mut().unwrap();
        assert_eq!(trap.next(), Some(20));
        assert_eq!(trap.next(), Some(30));
        assert_eq!(trap.next(), Some(10)); // wraps
    }

    #[test]
    fn test_focus_trap_release_restores_previous() {
        let mut fm = FocusManager::new();
        fm.focus(5);
        fm.push_trap(vec![10, 20]);
        fm.pop_trap();
        assert_eq!(fm.focused_node, Some(5));
    }

    #[test]
    fn test_tab_navigation_no_trap() {
        let tab_order = vec![1, 2, 3];
        let mut fm = FocusManager::new();
        fm.focus(1);
        let next = fm.tab(&tab_order, false);
        assert_eq!(next, Some(2));
        let next = fm.tab(&tab_order, false);
        assert_eq!(next, Some(3));
        let next = fm.tab(&tab_order, false);
        assert_eq!(next, Some(1)); // wrap
    }

    #[test]
    fn test_shift_tab_backwards() {
        let tab_order = vec![1, 2, 3];
        let mut fm = FocusManager::new();
        fm.focus(1);
        let prev = fm.tab(&tab_order, true);
        assert_eq!(prev, Some(3)); // wrap backwards
    }

    // ── Roving Tabindex ─────────────────────────────────────────────────────

    #[test]
    fn test_roving_tabindex_navigation() {
        let mut rt = RovingTabindex::new(vec![0, 1, 2, 3], true);
        assert_eq!(rt.current_item(), Some(0));
        assert_eq!(rt.next(), Some(1));
        assert_eq!(rt.next(), Some(2));
        assert_eq!(rt.prev(), Some(1));
        assert_eq!(rt.first(), Some(0));
        assert_eq!(rt.last(), Some(3));
    }

    #[test]
    fn test_roving_tabindex_wrap_forward() {
        let mut rt = RovingTabindex::new(vec![0, 1, 2], true);
        rt.last();
        assert_eq!(rt.next(), Some(0));
    }

    #[test]
    fn test_roving_tabindex_no_wrap() {
        let mut rt = RovingTabindex::new(vec![0, 1, 2], false);
        rt.last();
        // At last item; next() should stay at last with no wrap
        rt.next();
        assert_eq!(rt.current_item(), Some(2));
    }

    #[test]
    fn test_roving_tabindex_map() {
        let rt = RovingTabindex::new(vec![10, 20, 30], true);
        let map = rt.tabindex_map();
        assert_eq!(map[&10], 0);
        assert_eq!(map[&20], -1);
        assert_eq!(map[&30], -1);
    }

    // ── Announcement Queue ──────────────────────────────────────────────────

    #[test]
    fn test_assertive_clears_polite() {
        let mut q = AnnouncementQueue::new(Verbosity::High);
        q.announce("polite msg", LivePoliteness::Polite, false);
        q.announce("urgent!", LivePoliteness::Assertive, false);
        // Polite queue cleared by assertive
        assert_eq!(q.polite.len(), 0);
        let a = q.next().unwrap();
        assert_eq!(a.text, "urgent!");
    }

    #[test]
    fn test_polite_queued_in_order() {
        let mut q = AnnouncementQueue::new(Verbosity::Medium);
        q.announce("first", LivePoliteness::Polite, false);
        q.announce("second", LivePoliteness::Polite, false);
        assert_eq!(q.next().unwrap().text, "first");
        assert_eq!(q.next().unwrap().text, "second");
    }

    #[test]
    fn test_low_verbosity_ignores_polite() {
        let mut q = AnnouncementQueue::new(Verbosity::Low);
        q.announce("quiet", LivePoliteness::Polite, false);
        assert!(q.is_empty());
    }

    // ── Keyboard Shortcuts ──────────────────────────────────────────────────

    #[test]
    fn test_shortcut_registration_and_lookup() {
        let mut reg = ShortcutRegistry::new();
        reg.register(Shortcut {
            binding: KeyBinding::ctrl("s"),
            action: "save".into(),
            description: "Save file".into(),
            scope: "global".into(),
        }).unwrap();

        let found = reg.find(&KeyBinding::ctrl("s"));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].action, "save");
    }

    #[test]
    fn test_shortcut_conflict_same_scope() {
        let mut reg = ShortcutRegistry::new();
        reg.register(Shortcut {
            binding: KeyBinding::ctrl("s"),
            action: "save".into(),
            description: "Save".into(),
            scope: "global".into(),
        }).unwrap();

        let result = reg.register(Shortcut {
            binding: KeyBinding::ctrl("s"),
            action: "search".into(),
            description: "Search".into(),
            scope: "global".into(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_shortcut_same_key_different_scope() {
        let mut reg = ShortcutRegistry::new();
        reg.register(Shortcut {
            binding: KeyBinding::ctrl("s"),
            action: "save".into(),
            description: "Save".into(),
            scope: "editor".into(),
        }).unwrap();
        // Same key, different scope — should succeed
        let result = reg.register(Shortcut {
            binding: KeyBinding::ctrl("s"),
            action: "submit".into(),
            description: "Submit".into(),
            scope: "dialog".into(),
        });
        assert!(result.is_ok());
    }

    // ── Accessibility Audit ─────────────────────────────────────────────────

    #[test]
    fn test_audit_missing_img_alt() {
        let mut tree = AXTree::new();
        let img = AXNode::new(0, "img", AriaRole::Img);
        // No alt (text_content empty, no aria-label)
        tree.add_node(img);
        tree.set_root(0);
        tree.compute_names();

        let violations = AccessibilityAuditor::audit(&tree, &HashMap::new(), &HashMap::new());
        let sc111: Vec<_> = violations.iter().filter(|v| v.criterion == "1.1.1").collect();
        assert!(!sc111.is_empty(), "should flag missing alt for img");
    }

    #[test]
    fn test_audit_low_contrast() {
        let mut tree = AXTree::new();
        let div = AXNode::new(0, "div", AriaRole::None);
        tree.add_node(div);
        tree.set_root(0);
        tree.compute_names();

        let mut fg_map = HashMap::new();
        let mut bg_map = HashMap::new();
        // Light grey on white — fails contrast
        fg_map.insert(0, Color::rgb(200, 200, 200));
        bg_map.insert(0, Color::rgb(255, 255, 255));

        let violations = AccessibilityAuditor::audit(&tree, &fg_map, &bg_map);
        let sc143: Vec<_> = violations.iter().filter(|v| v.criterion == "1.4.3").collect();
        assert!(!sc143.is_empty(), "should flag low contrast");
    }

    #[test]
    fn test_audit_no_violations_accessible_button() {
        let mut tree = AXTree::new();
        let mut btn = AXNode::new(0, "button", AriaRole::Button);
        btn.text_content = "Submit".into();
        btn.tab_index = Some(0);
        btn.bounds = BoundingBox::new(0.0, 0.0, 120.0, 44.0);
        tree.add_node(btn);
        tree.set_root(0);
        tree.compute_names();

        let mut fg_map = HashMap::new();
        let mut bg_map = HashMap::new();
        fg_map.insert(0, Color::rgb(0, 0, 0));
        bg_map.insert(0, Color::rgb(255, 255, 255));

        let violations = AccessibilityAuditor::audit(&tree, &fg_map, &bg_map);
        // No violations expected
        assert!(violations.is_empty(), "accessible button should produce no violations: {:?}", violations);
    }

    #[test]
    fn test_audit_touch_target_too_small() {
        let mut tree = AXTree::new();
        let mut btn = AXNode::new(0, "button", AriaRole::Button);
        btn.text_content = "X".into();
        btn.bounds = BoundingBox::new(0.0, 0.0, 20.0, 20.0);
        btn.tab_index = Some(0);
        tree.add_node(btn);
        tree.set_root(0);
        tree.compute_names();

        let violations = AccessibilityAuditor::audit(&tree, &HashMap::new(), &HashMap::new());
        let sc255: Vec<_> = violations.iter().filter(|v| v.criterion == "2.5.5").collect();
        assert!(!sc255.is_empty(), "should flag small touch target");
    }

    // ── Landmark & Heading Navigation ───────────────────────────────────────

    #[test]
    fn test_landmark_navigator() {
        let mut tree = AXTree::new();
        let nav = AXNode::new(0, "nav", AriaRole::Navigation);
        let main = AXNode::new(1, "main", AriaRole::Main);
        let aside = AXNode::new(2, "aside", AriaRole::Complementary);
        let nav_id = tree.add_node(nav);
        let main_id = tree.add_node(main);
        let aside_id = tree.add_node(aside);
        tree.set_root(nav_id);
        tree.add_child(nav_id, main_id);
        tree.add_child(main_id, aside_id);

        let mut ln = LandmarkNavigator::from_tree(&tree);
        assert_eq!(ln.count(), 3);
        let next = ln.next_landmark().unwrap();
        assert!(next == main_id || next == nav_id || next == aside_id);
    }

    #[test]
    fn test_heading_navigator_outline() {
        let mut tree = AXTree::new();
        let h1 = { let mut n = AXNode::new(0, "h1", AriaRole::Heading); n.props.level = Some(1); n };
        let h2 = { let mut n = AXNode::new(1, "h2", AriaRole::Heading); n.props.level = Some(2); n };
        let h3 = { let mut n = AXNode::new(2, "h3", AriaRole::Heading); n.props.level = Some(3); n };
        let id0 = tree.add_node(h1);
        let id1 = tree.add_node(h2);
        let id2 = tree.add_node(h3);
        tree.set_root(id0);
        tree.add_child(id0, id1);
        tree.add_child(id1, id2);

        let hn = HeadingNavigator::from_tree(&tree);
        assert_eq!(hn.count(), 3);
        let outline = hn.outline();
        assert_eq!(outline[0], (1, id0));
        assert_eq!(outline[1], (2, id1));
        assert_eq!(outline[2], (3, id2));
    }

    // ── Utility ─────────────────────────────────────────────────────────────

    #[test]
    fn test_strip_html() {
        let input = "<strong>Hello</strong> <em>world</em>!";
        assert_eq!(strip_html(input), "Hello world!");
    }

    #[test]
    fn test_normalise_whitespace() {
        let s = "  foo   bar\tbaz  ";
        assert_eq!(normalise_whitespace(s), "foo bar baz");
    }

    #[test]
    fn test_animation_duration_reduced_motion() {
        let dur = animation_duration(300.0, &MotionPreference::Reduce);
        assert!(dur <= 1.0);
    }

    #[test]
    fn test_animation_duration_no_preference() {
        let dur = animation_duration(300.0, &MotionPreference::NoPreference);
        assert!((dur - 300.0).abs() < 1e-9);
    }

    #[test]
    fn test_bounding_box_touch_target() {
        assert!(BoundingBox::new(0.0, 0.0, 44.0, 44.0).meets_touch_target());
        assert!(!BoundingBox::new(0.0, 0.0, 43.0, 44.0).meets_touch_target());
        assert!(!BoundingBox::new(0.0, 0.0, 44.0, 43.0).meets_touch_target());
    }

    #[test]
    fn test_focus_trap_prev_wraps() {
        let mut trap = FocusTrap::new(vec![1, 2, 3], None);
        // At index 0, prev() should wrap to last
        let prev = trap.prev();
        assert_eq!(prev, Some(3));
    }

    #[test]
    fn test_audit_summary_groups_by_criterion() {
        let violations = vec![
            Violation { criterion: "1.1.1".into(), impact: Impact::Critical, node_id: 0,
                message: "".into(), help_url: "".into() },
            Violation { criterion: "1.1.1".into(), impact: Impact::Critical, node_id: 1,
                message: "".into(), help_url: "".into() },
            Violation { criterion: "1.4.3".into(), impact: Impact::Serious, node_id: 2,
                message: "".into(), help_url: "".into() },
        ];
        let summary = AccessibilityAuditor::summary(&violations);
        assert_eq!(summary["1.1.1"], 2);
        assert_eq!(summary["1.4.3"], 1);
    }
}
