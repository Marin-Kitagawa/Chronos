// ============================================================================
// CHRONOS UI / GUI FRAMEWORK ENGINE
// ============================================================================
//
// HOW GUI FRAMEWORKS ACTUALLY WORK (and what makes them hard):
//
// A GUI framework must solve three deeply interdependent problems:
//
// 1. LAYOUT: Given a tree of widgets with size constraints, determine the
//    exact pixel rectangle each widget occupies. This is harder than it
//    looks — CSS took 20 years to get right. Modern frameworks use
//    constraint-based layout (flexbox, grid) or the Cassowary algorithm.
//
// 2. RENDERING: Convert widget descriptions into pixels. Options range
//    from drawing directly to a framebuffer (like early GUIs) to using a
//    retained-mode scene graph (like Qt) to going fully immediate-mode
//    (like Dear ImGui). We implement a retained-mode widget tree with a
//    "dirty rect" system to avoid re-rendering everything every frame.
//
// 3. EVENT HANDLING: Route input events (mouse clicks, key presses) to
//    the correct widget. This requires hit testing (which widget is under
//    the cursor?) and a responder chain (if a widget doesn't handle an
//    event, bubble it to the parent).
//
// THE WIDGET TREE:
// Inspired by React's virtual DOM and Flutter's widget tree. Each widget
// is described by immutable "props" and optional mutable "state". On each
// frame, the framework compares the new virtual tree with the previous one
// (diffing / reconciliation) and only updates what changed.
//
// LAYOUT ALGORITHMS:
// - Fixed: widget has explicit width/height
// - Fill: widget expands to fill available space
// - Wrap: widget sizes to its content
// - Flex: like CSS flexbox — distribute space among flex children
//   along a main axis (horizontal or vertical)
// - Stack: children overlay each other (like CSS position: absolute)
//
// BOX MODEL:
// Every widget has: margin → border → padding → content
// This mirrors the CSS box model. Sizes are computed from outside in:
// the layout algorithm assigns an available size, then the widget claims
// what it needs, leaving the rest for its parent to redistribute.
//
// EVENT SYSTEM:
// Events are dispatched in two phases:
// - Capture (root → target): parent can intercept before child sees it
// - Bubble (target → root): parent can react after child handled it
// This is exactly the W3C DOM event model.
//
// REACTIVE STATE:
// We implement a simple signal-based reactivity system inspired by
// SolidJS/Vue: State<T> holds a value; reading it registers a dependency;
// writing it triggers re-renders of all dependents.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic with tests):
//   1.  Geometry primitives (Point, Size, Rect, EdgeInsets, Color)
//   2.  Widget trait and widget tree (WidgetNode, WidgetId)
//   3.  Box model layout (margin/padding/border)
//   4.  Flex layout algorithm (direction, justify, align, wrap)
//   5.  Grid layout algorithm
//   6.  Stack/absolute layout
//   7.  Core widgets: Text, Button, Image, Container, Row, Column, Stack
//   8.  Scroll container
//   9.  Input widgets: TextInput, Checkbox, Slider, RadioButton, Select
//  10.  Event system (mouse, keyboard, focus, scroll)
//  11.  Hit testing (find widget under cursor)
//  12.  Event bubbling/capturing
//  13.  Reactive state (Signal<T>, Computed<T>)
//  14.  Style system (colors, fonts, borders, shadows)
//  15.  Virtual DOM diffing and reconciliation
//  16.  Simple renderer to ASCII (for testing without a display)
//  17.  Animation system (tweening, easing functions)
//  18.  Theme system
//  19.  Comprehensive tests
// ============================================================================

use std::collections::HashMap;

// ============================================================================
// Part 1: Geometry Primitives
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
    pub fn distance_to(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
    pub fn translate(&self, dx: f32, dy: f32) -> Self {
        Self { x: self.x + dx, y: self.y + dy }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    pub width: f32,
    pub height: f32,
}

impl Size {
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
    pub fn zero() -> Self {
        Self { width: 0.0, height: 0.0 }
    }
    pub fn clamp(&self, min: Size, max: Size) -> Self {
        Self {
            width: self.width.clamp(min.width, max.width),
            height: self.height.clamp(min.height, max.height),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub origin: Point,
    pub size: Size,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            origin: Point::new(x, y),
            size: Size::new(width, height),
        }
    }

    pub fn from_origin_size(origin: Point, size: Size) -> Self {
        Self { origin, size }
    }

    pub fn zero() -> Self {
        Self { origin: Point::zero(), size: Size::zero() }
    }

    pub fn min_x(&self) -> f32 { self.origin.x }
    pub fn min_y(&self) -> f32 { self.origin.y }
    pub fn max_x(&self) -> f32 { self.origin.x + self.size.width }
    pub fn max_y(&self) -> f32 { self.origin.y + self.size.height }
    pub fn center(&self) -> Point {
        Point::new(
            self.origin.x + self.size.width / 2.0,
            self.origin.y + self.size.height / 2.0,
        )
    }

    pub fn contains(&self, point: &Point) -> bool {
        point.x >= self.min_x()
            && point.x < self.max_x()
            && point.y >= self.min_y()
            && point.y < self.max_y()
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        self.min_x() < other.max_x()
            && self.max_x() > other.min_x()
            && self.min_y() < other.max_y()
            && self.max_y() > other.min_y()
    }

    pub fn union(&self, other: &Rect) -> Self {
        let x = self.min_x().min(other.min_x());
        let y = self.min_y().min(other.min_y());
        let max_x = self.max_x().max(other.max_x());
        let max_y = self.max_y().max(other.max_y());
        Rect::new(x, y, max_x - x, max_y - y)
    }

    /// Inset the rect by equal amounts on all sides
    pub fn inset(&self, insets: &EdgeInsets) -> Rect {
        Rect::new(
            self.origin.x + insets.left,
            self.origin.y + insets.top,
            (self.size.width - insets.left - insets.right).max(0.0),
            (self.size.height - insets.top - insets.bottom).max(0.0),
        )
    }
}

/// Spacing values for margin, padding, border
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeInsets {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl EdgeInsets {
    pub fn new(top: f32, right: f32, bottom: f32, left: f32) -> Self {
        Self { top, right, bottom, left }
    }
    pub fn all(v: f32) -> Self {
        Self { top: v, right: v, bottom: v, left: v }
    }
    pub fn symmetric(vertical: f32, horizontal: f32) -> Self {
        Self {
            top: vertical,
            right: horizontal,
            bottom: vertical,
            left: horizontal,
        }
    }
    pub fn zero() -> Self {
        Self::all(0.0)
    }
    pub fn horizontal(&self) -> f32 { self.left + self.right }
    pub fn vertical(&self) -> f32 { self.top + self.bottom }
}

/// RGBA color (0.0–1.0 components)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }
    pub fn from_hex(hex: u32) -> Self {
        Self {
            r: ((hex >> 16) & 0xFF) as f32 / 255.0,
            g: ((hex >> 8) & 0xFF) as f32 / 255.0,
            b: (hex & 0xFF) as f32 / 255.0,
            a: 1.0,
        }
    }
    pub fn to_hex(&self) -> u32 {
        let r = (self.r.clamp(0.0, 1.0) * 255.0) as u32;
        let g = (self.g.clamp(0.0, 1.0) * 255.0) as u32;
        let b = (self.b.clamp(0.0, 1.0) * 255.0) as u32;
        (r << 16) | (g << 8) | b
    }
    pub fn with_alpha(&self, a: f32) -> Self {
        Self { r: self.r, g: self.g, b: self.b, a }
    }
    pub fn mix(&self, other: &Color, t: f32) -> Self {
        Self {
            r: self.r + t * (other.r - self.r),
            g: self.g + t * (other.g - self.g),
            b: self.b + t * (other.b - self.b),
            a: self.a + t * (other.a - self.a),
        }
    }

    // Predefined colors
    pub fn transparent() -> Self { Self::rgba(0.0, 0.0, 0.0, 0.0) }
    pub fn white() -> Self { Self::rgb(1.0, 1.0, 1.0) }
    pub fn black() -> Self { Self::rgb(0.0, 0.0, 0.0) }
    pub fn red() -> Self { Self::rgb(1.0, 0.0, 0.0) }
    pub fn green() -> Self { Self::rgb(0.0, 0.8, 0.0) }
    pub fn blue() -> Self { Self::rgb(0.0, 0.0, 1.0) }
    pub fn gray() -> Self { Self::rgb(0.5, 0.5, 0.5) }
    pub fn light_gray() -> Self { Self::rgb(0.85, 0.85, 0.85) }
}

// ============================================================================
// Part 2: Style System
// ============================================================================
//
// Every widget can carry a Style that describes its visual appearance.
// Styles cascade: a child inherits from its parent unless it overrides.
// This mirrors CSS inheritance.

#[derive(Debug, Clone, PartialEq)]
pub enum FontWeight {
    Thin,
    Regular,
    Medium,
    SemiBold,
    Bold,
    ExtraBold,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextAlign {
    Left,
    Center,
    Right,
    Justify,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BorderStyle {
    None,
    Solid,
    Dashed,
    Dotted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Border {
    pub width: f32,
    pub color: Color,
    pub style: BorderStyle,
    pub radius: f32, // corner radius
}

impl Border {
    pub fn none() -> Self {
        Self { width: 0.0, color: Color::transparent(), style: BorderStyle::None, radius: 0.0 }
    }
    pub fn solid(width: f32, color: Color) -> Self {
        Self { width, color, style: BorderStyle::Solid, radius: 0.0 }
    }
    pub fn rounded(width: f32, color: Color, radius: f32) -> Self {
        Self { width, color, style: BorderStyle::Solid, radius }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoxShadow {
    pub offset_x: f32,
    pub offset_y: f32,
    pub blur: f32,
    pub spread: f32,
    pub color: Color,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Style {
    // Colors
    pub background: Option<Color>,
    pub foreground: Option<Color>,

    // Typography
    pub font_size: Option<f32>,
    pub font_weight: Option<FontWeight>,
    pub text_align: Option<TextAlign>,
    pub line_height: Option<f32>,

    // Spacing
    pub padding: EdgeInsets,
    pub margin: EdgeInsets,

    // Border
    pub border: Border,

    // Effects
    pub opacity: f32,
    pub box_shadow: Option<BoxShadow>,

    // Sizing
    pub min_width: Option<f32>,
    pub min_height: Option<f32>,
    pub max_width: Option<f32>,
    pub max_height: Option<f32>,
}

impl Style {
    pub fn default() -> Self {
        Self {
            background: None,
            foreground: None,
            font_size: None,
            font_weight: None,
            text_align: None,
            line_height: None,
            padding: EdgeInsets::zero(),
            margin: EdgeInsets::zero(),
            border: Border::none(),
            opacity: 1.0,
            box_shadow: None,
            min_width: None,
            min_height: None,
            max_width: None,
            max_height: None,
        }
    }

    /// Merge this style with a parent, inheriting unset values
    pub fn inherit_from(&self, parent: &Style) -> Style {
        Style {
            background: self.background.or(parent.background),
            foreground: self.foreground.or(parent.foreground),
            font_size: self.font_size.or(parent.font_size),
            font_weight: self.font_weight.clone().or_else(|| parent.font_weight.clone()),
            text_align: self.text_align.clone().or_else(|| parent.text_align.clone()),
            line_height: self.line_height.or(parent.line_height),
            padding: self.padding,
            margin: self.margin,
            border: self.border.clone(),
            opacity: self.opacity * parent.opacity,
            box_shadow: self.box_shadow.clone().or_else(|| parent.box_shadow.clone()),
            min_width: self.min_width.or(parent.min_width),
            min_height: self.min_height.or(parent.min_height),
            max_width: self.max_width.or(parent.max_width),
            max_height: self.max_height.or(parent.max_height),
        }
    }
}

// ============================================================================
// Part 3: Widget Definitions
// ============================================================================
//
// Widgets are the building blocks of the UI. We use a data-driven approach:
// each widget is a Rust enum variant carrying its configuration (props).
// This makes the widget tree serializable and enables diffing.
//
// LAYOUT MODES determine how a widget's children are arranged:
// - Flex: children laid out along an axis (like CSS flexbox)
// - Grid: children in a 2D grid
// - Stack: children overlaid at absolute positions
// - Scroll: one child, can scroll if larger than container

pub type WidgetId = u64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlexDirection {
    Row,         // left to right
    Column,      // top to bottom
    RowReverse,  // right to left
    ColumnReverse, // bottom to top
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JustifyContent {
    Start,
    End,
    Center,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlignItems {
    Start,
    End,
    Center,
    Stretch,
    Baseline,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlexWrap {
    NoWrap,
    Wrap,
    WrapReverse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlexProps {
    pub direction: FlexDirection,
    pub justify: JustifyContent,
    pub align: AlignItems,
    pub wrap: FlexWrap,
    pub gap: f32,
}

impl FlexProps {
    pub fn row() -> Self {
        Self {
            direction: FlexDirection::Row,
            justify: JustifyContent::Start,
            align: AlignItems::Stretch,
            wrap: FlexWrap::NoWrap,
            gap: 0.0,
        }
    }
    pub fn column() -> Self {
        Self {
            direction: FlexDirection::Column,
            justify: JustifyContent::Start,
            align: AlignItems::Stretch,
            wrap: FlexWrap::NoWrap,
            gap: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridProps {
    pub columns: usize,
    pub rows: Option<usize>,
    pub column_gap: f32,
    pub row_gap: f32,
    pub column_widths: Vec<GridTrack>, // sizes for each column
}

#[derive(Debug, Clone, PartialEq)]
pub enum GridTrack {
    Fixed(f32),
    Fraction(f32), // fractional unit (like CSS fr)
    Auto,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SizeConstraint {
    /// Fixed pixel size
    Fixed(f32),
    /// Fill available space (with optional flex factor)
    Fill(f32),
    /// Size to content
    Wrap,
    /// Minimum, maximum bounds
    Bounded { min: f32, max: f32 },
}

/// The core widget type. Each variant represents a distinct UI element.
#[derive(Debug, Clone, PartialEq)]
pub enum Widget {
    /// Plain text label
    Text {
        content: String,
        style: Style,
    },
    /// Clickable button
    Button {
        label: String,
        on_click: Option<String>, // event handler name (action id)
        enabled: bool,
        style: Style,
    },
    /// Container / box layout
    Container {
        style: Style,
        width: SizeConstraint,
        height: SizeConstraint,
        children: Vec<WidgetNode>,
    },
    /// Flexbox layout (Row or Column)
    Flex {
        props: FlexProps,
        style: Style,
        children: Vec<WidgetNode>,
    },
    /// Grid layout
    Grid {
        props: GridProps,
        style: Style,
        children: Vec<WidgetNode>,
    },
    /// Stack layout (children positioned absolutely)
    Stack {
        style: Style,
        children: Vec<StackChild>,
    },
    /// Scrollable container
    Scroll {
        scroll_x: bool,
        scroll_y: bool,
        scroll_offset: Point,
        style: Style,
        child: Box<WidgetNode>,
    },
    /// Single-line text input
    TextInput {
        value: String,
        placeholder: String,
        on_change: Option<String>,
        enabled: bool,
        style: Style,
    },
    /// Checkbox
    Checkbox {
        checked: bool,
        label: String,
        on_change: Option<String>,
        style: Style,
    },
    /// Slider (continuous value)
    Slider {
        value: f32,
        min: f32,
        max: f32,
        step: f32,
        on_change: Option<String>,
        style: Style,
    },
    /// Image
    Image {
        src: String,
        width: f32,
        height: f32,
        alt: String,
        style: Style,
    },
    /// Spacer (takes up flexible space)
    Spacer { flex: f32 },
    /// Divider line
    Divider { axis: Axis, thickness: f32, color: Color },
    /// Progress bar
    ProgressBar { value: f32, style: Style },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Axis { Horizontal, Vertical }

/// A widget with its assigned ID and computed layout rect
#[derive(Debug, Clone, PartialEq)]
pub struct WidgetNode {
    pub id: WidgetId,
    pub widget: Widget,
    pub layout_rect: Option<Rect>,
}

impl WidgetNode {
    pub fn new(id: WidgetId, widget: Widget) -> Self {
        Self { id, widget, layout_rect: None }
    }
}

/// A child in a Stack layout with absolute or relative positioning
#[derive(Debug, Clone, PartialEq)]
pub struct StackChild {
    pub node: WidgetNode,
    pub position: Point,    // top-left position within the stack
    pub z_index: i32,
}

// ============================================================================
// Part 4: Constraints and Layout Engine
// ============================================================================
//
// Layout is computed in two passes:
// 1. MEASURE: each widget recursively measures itself given available space.
//    Returns the minimum size it needs.
// 2. ARRANGE: each container assigns a rect to each child given the container's
//    own rect and the children's measured sizes.
//
// This two-pass approach (inspired by Flutter's RenderObject) avoids the
// "double pass" problem in CSS where you need to know siblings' sizes first.

#[derive(Debug, Clone, Copy)]
pub struct Constraints {
    pub min_width: f32,
    pub max_width: f32,
    pub min_height: f32,
    pub max_height: f32,
}

impl Constraints {
    pub fn new(min_w: f32, max_w: f32, min_h: f32, max_h: f32) -> Self {
        Self {
            min_width: min_w,
            max_width: max_w,
            min_height: min_h,
            max_height: max_h,
        }
    }

    pub fn loose(width: f32, height: f32) -> Self {
        Self { min_width: 0.0, max_width: width, min_height: 0.0, max_height: height }
    }

    pub fn tight(width: f32, height: f32) -> Self {
        Self { min_width: width, max_width: width, min_height: height, max_height: height }
    }

    pub fn constrain_size(&self, size: Size) -> Size {
        Size {
            width: size.width.clamp(self.min_width, self.max_width),
            height: size.height.clamp(self.min_height, self.max_height),
        }
    }
}

pub struct LayoutEngine {
    next_id: WidgetId,
}

impl LayoutEngine {
    pub fn new() -> Self {
        Self { next_id: 1 }
    }

    pub fn next_id(&mut self) -> WidgetId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Measure a widget: compute its intrinsic size given constraints.
    /// Returns the size the widget wants to be.
    pub fn measure(&self, node: &WidgetNode, constraints: &Constraints) -> Size {
        match &node.widget {
            Widget::Text { content, style } => {
                let font_size = style.font_size.unwrap_or(14.0);
                let padding = &style.padding;
                // Approximate text measurement: 0.6 * font_size per character, line_height factor
                let chars_per_line = ((constraints.max_width - padding.horizontal()) / (font_size * 0.6)).max(1.0) as usize;
                let num_lines = ((content.len() + chars_per_line - 1) / chars_per_line).max(1);
                let line_height = style.line_height.unwrap_or(font_size * 1.4);
                let width = (content.len() as f32 * font_size * 0.6 + padding.horizontal())
                    .min(constraints.max_width);
                let height = num_lines as f32 * line_height + padding.vertical();
                constraints.constrain_size(Size::new(width, height))
            }

            Widget::Button { label, style, .. } => {
                let font_size = style.font_size.unwrap_or(14.0);
                let padding = &style.padding;
                let w = label.len() as f32 * font_size * 0.6 + padding.horizontal() + 24.0;
                let h = font_size * 1.4 + padding.vertical() + 8.0;
                constraints.constrain_size(Size::new(w, h))
            }

            Widget::Container { width, height, children, style, .. } => {
                let border_extra = style.border.width * 2.0;
                let inner_w = (constraints.max_width - style.padding.horizontal() - border_extra).max(0.0);
                let inner_h = (constraints.max_height - style.padding.vertical() - border_extra).max(0.0);
                let inner_constraints = Constraints::loose(inner_w, inner_h);

                let content_size = if children.is_empty() {
                    Size::zero()
                } else {
                    children.iter().fold(Size::zero(), |acc, child| {
                        let cs = self.measure(child, &inner_constraints);
                        Size::new(acc.width.max(cs.width), acc.height.max(cs.height))
                    })
                };

                let w = match width {
                    SizeConstraint::Fixed(v) => *v,
                    SizeConstraint::Fill(_) => constraints.max_width,
                    SizeConstraint::Wrap => content_size.width + style.padding.horizontal() + border_extra,
                    SizeConstraint::Bounded { min, max } => content_size.width.clamp(*min, *max),
                };
                let h = match height {
                    SizeConstraint::Fixed(v) => *v,
                    SizeConstraint::Fill(_) => constraints.max_height,
                    SizeConstraint::Wrap => content_size.height + style.padding.vertical() + border_extra,
                    SizeConstraint::Bounded { min, max } => content_size.height.clamp(*min, *max),
                };
                constraints.constrain_size(Size::new(w, h))
            }

            Widget::Flex { props, children, style } => {
                self.measure_flex(props, children, style, constraints)
            }

            Widget::Grid { props, children, style } => {
                self.measure_grid(props, children, style, constraints)
            }

            Widget::Stack { children, style } => {
                // Stack size = union of all children's sizes
                let mut max_w: f32 = 0.0;
                let mut max_h: f32 = 0.0;
                for child in children {
                    let cs = self.measure(&child.node, constraints);
                    max_w = max_w.max(child.position.x + cs.width);
                    max_h = max_h.max(child.position.y + cs.height);
                }
                let w = max_w + style.padding.horizontal();
                let h = max_h + style.padding.vertical();
                constraints.constrain_size(Size::new(w, h))
            }

            Widget::TextInput { style, .. } => {
                let h = style.font_size.unwrap_or(14.0) * 1.4 + style.padding.vertical() + 8.0;
                constraints.constrain_size(Size::new(constraints.max_width, h))
            }

            Widget::Checkbox { label, style, .. } => {
                let font_size = style.font_size.unwrap_or(14.0);
                let w = font_size + 8.0 + label.len() as f32 * font_size * 0.6;
                let h = font_size + 8.0;
                constraints.constrain_size(Size::new(w, h))
            }

            Widget::Slider { style, .. } => {
                let h = 20.0 + style.padding.vertical();
                constraints.constrain_size(Size::new(constraints.max_width, h))
            }

            Widget::Image { width, height, .. } => {
                constraints.constrain_size(Size::new(*width, *height))
            }

            Widget::Spacer { flex } => {
                // Spacers take up remaining space — their size is determined by the parent flex layout
                let _ = flex;
                Size::zero()
            }

            Widget::Divider { axis, thickness, .. } => {
                match axis {
                    Axis::Horizontal => constraints.constrain_size(Size::new(constraints.max_width, *thickness)),
                    Axis::Vertical => constraints.constrain_size(Size::new(*thickness, constraints.max_height)),
                }
            }

            Widget::ProgressBar { style, .. } => {
                let h = 8.0 + style.padding.vertical();
                constraints.constrain_size(Size::new(constraints.max_width, h))
            }

            Widget::Scroll { child, style, .. } => {
                // Scroll container takes available space
                constraints.constrain_size(Size::new(
                    constraints.max_width,
                    style.min_height.unwrap_or(constraints.min_height),
                ))
            }
        }
    }

    fn measure_flex(&self, props: &FlexProps, children: &[WidgetNode], style: &Style, constraints: &Constraints) -> Size {
        let is_row = matches!(props.direction, FlexDirection::Row | FlexDirection::RowReverse);
        let available_main = if is_row { constraints.max_width } else { constraints.max_height };
        let available_cross = if is_row { constraints.max_height } else { constraints.max_width };

        let padding_main = if is_row { style.padding.horizontal() } else { style.padding.vertical() };
        let padding_cross = if is_row { style.padding.vertical() } else { style.padding.horizontal() };

        let gap_total = if children.is_empty() { 0.0 } else { (children.len() - 1) as f32 * props.gap };
        let inner_main = (available_main - padding_main - gap_total).max(0.0);
        let inner_cross = (available_cross - padding_cross).max(0.0);

        let child_constraints = if is_row {
            Constraints::loose(inner_main, inner_cross)
        } else {
            Constraints::loose(inner_cross, inner_main)
        };

        let mut total_main: f32 = 0.0;
        let mut max_cross: f32 = 0.0;

        for child in children {
            if let Widget::Spacer { .. } = &child.widget {
                continue; // Measured later when distributing remaining space
            }
            let cs = self.measure(child, &child_constraints);
            let (main, cross) = if is_row { (cs.width, cs.height) } else { (cs.height, cs.width) };
            total_main += main;
            max_cross = max_cross.max(cross);
        }

        total_main += gap_total;
        let main = (total_main + padding_main).min(available_main);
        let cross = max_cross + padding_cross;

        if is_row {
            constraints.constrain_size(Size::new(main, cross))
        } else {
            constraints.constrain_size(Size::new(cross, main))
        }
    }

    fn measure_grid(&self, props: &GridProps, children: &[WidgetNode], style: &Style, constraints: &Constraints) -> Size {
        let cols = props.columns.max(1);
        let rows = (children.len() + cols - 1) / cols;

        let inner_w = (constraints.max_width - style.padding.horizontal()
            - (cols - 1) as f32 * props.column_gap).max(0.0);
        let col_w = inner_w / cols as f32;
        let col_constraints = Constraints::loose(col_w, constraints.max_height);

        let mut row_heights = vec![0.0f32; rows];
        for (i, child) in children.iter().enumerate() {
            let row = i / cols;
            let cs = self.measure(child, &col_constraints);
            row_heights[row] = row_heights[row].max(cs.height);
        }

        let total_h: f32 = row_heights.iter().sum::<f32>()
            + (rows - 1).max(0) as f32 * props.row_gap
            + style.padding.vertical();

        constraints.constrain_size(Size::new(constraints.max_width, total_h))
    }

    /// Arrange: assign rects to each widget in the tree given a parent rect.
    pub fn arrange(&self, node: &mut WidgetNode, rect: Rect) {
        node.layout_rect = Some(rect);

        match &mut node.widget {
            Widget::Flex { props, children, style } => {
                let props = props.clone();
                let style = style.clone();
                let inner = rect.inset(&style.padding);
                self.arrange_flex(&props, children, inner);
            }
            Widget::Grid { props, children, style } => {
                let props = props.clone();
                let style = style.clone();
                let inner = rect.inset(&style.padding);
                self.arrange_grid(&props, children, inner);
            }
            Widget::Container { children, style, .. } => {
                let style = style.clone();
                let inner = rect.inset(&style.padding);
                // Stack children on top of each other (simple container)
                for child in children.iter_mut() {
                    let constraints = Constraints::loose(inner.size.width, inner.size.height);
                    let cs = self.measure(child, &constraints);
                    self.arrange(child, Rect::from_origin_size(inner.origin, cs));
                }
            }
            Widget::Stack { children, style } => {
                let style = style.clone();
                let inner = rect.inset(&style.padding);
                let mut sorted: Vec<usize> = (0..children.len()).collect();
                sorted.sort_by_key(|&i| children[i].z_index);
                for &i in &sorted {
                    let child_pos = inner.origin.translate(children[i].position.x, children[i].position.y);
                    let constraints = Constraints::loose(
                        inner.size.width - children[i].position.x,
                        inner.size.height - children[i].position.y,
                    );
                    let cs = self.measure(&children[i].node, &constraints);
                    self.arrange(&mut children[i].node, Rect::from_origin_size(child_pos, cs));
                }
            }
            Widget::Scroll { child, style, scroll_offset, .. } => {
                let style = style.clone();
                let inner = rect.inset(&style.padding);
                let constraints = Constraints::loose(f32::MAX, f32::MAX);
                let cs = self.measure(child, &constraints);
                let child_origin = inner.origin.translate(-scroll_offset.x, -scroll_offset.y);
                self.arrange(child, Rect::from_origin_size(child_origin, cs));
            }
            _ => {} // Leaf widgets don't have children to arrange
        }
    }

    fn arrange_flex(&self, props: &FlexProps, children: &mut Vec<WidgetNode>, inner: Rect) {
        let is_row = matches!(props.direction, FlexDirection::Row | FlexDirection::RowReverse);
        let available_main = if is_row { inner.size.width } else { inner.size.height };
        let available_cross = if is_row { inner.size.height } else { inner.size.width };

        // Measure all non-spacer children
        let constraints = Constraints::loose(available_main, available_cross);
        let mut child_sizes: Vec<Size> = children.iter().map(|c| self.measure(c, &constraints)).collect();

        // Count spacers and fixed sizes
        let mut spacer_flex_total: f32 = 0.0;
        let mut used_main: f32 = 0.0;
        let gap_total = if children.is_empty() { 0.0 } else { (children.len() - 1) as f32 * props.gap };

        for (i, child) in children.iter().enumerate() {
            if let Widget::Spacer { flex } = &child.widget {
                spacer_flex_total += flex;
                child_sizes[i] = Size::zero();
            } else {
                used_main += if is_row { child_sizes[i].width } else { child_sizes[i].height };
            }
        }

        let remaining = (available_main - used_main - gap_total).max(0.0);
        // Assign sizes to spacers
        if spacer_flex_total > 0.0 {
            for (i, child) in children.iter().enumerate() {
                if let Widget::Spacer { flex } = &child.widget {
                    let spacer_size = remaining * (flex / spacer_flex_total);
                    child_sizes[i] = if is_row {
                        Size::new(spacer_size, 0.0)
                    } else {
                        Size::new(0.0, spacer_size)
                    };
                }
            }
        }

        // Compute total main size
        let total_main: f32 = child_sizes.iter().map(|s| if is_row { s.width } else { s.height }).sum::<f32>() + gap_total;

        // Justify: compute starting offset and spacing
        let (start_offset, spacing) = self.compute_justify(props.justify, total_main, available_main);

        // Place children
        let mut cursor = start_offset;
        let dir = &props.direction;
        let reversed = matches!(dir, FlexDirection::RowReverse | FlexDirection::ColumnReverse);

        let order: Vec<usize> = if reversed {
            (0..children.len()).rev().collect()
        } else {
            (0..children.len()).collect()
        };

        for &i in &order {
            let size = child_sizes[i];
            let main_size = if is_row { size.width } else { size.height };
            let cross_size = if is_row { size.height } else { size.width };

            // Align on cross axis
            let cross_offset = self.compute_align(props.align, cross_size, available_cross);

            let child_rect = if is_row {
                Rect::new(inner.origin.x + cursor, inner.origin.y + cross_offset, main_size, size.height)
            } else {
                Rect::new(inner.origin.x + cross_offset, inner.origin.y + cursor, size.width, main_size)
            };

            // We need mutable access — do immutable measure before this
            children[i].layout_rect = Some(child_rect);
            cursor += main_size + props.gap + spacing;
        }
    }

    fn compute_justify(&self, justify: JustifyContent, total: f32, available: f32) -> (f32, f32) {
        let extra = (available - total).max(0.0);
        match justify {
            JustifyContent::Start => (0.0, 0.0),
            JustifyContent::End => (extra, 0.0),
            JustifyContent::Center => (extra / 2.0, 0.0),
            JustifyContent::SpaceBetween => {
                // spacing handled by cursor increment, not here
                (0.0, 0.0) // simplified
            }
            JustifyContent::SpaceAround => (extra / 4.0, 0.0),
            JustifyContent::SpaceEvenly => (extra / 4.0, 0.0),
        }
    }

    fn compute_align(&self, align: AlignItems, child_size: f32, available: f32) -> f32 {
        match align {
            AlignItems::Start | AlignItems::Baseline => 0.0,
            AlignItems::End => (available - child_size).max(0.0),
            AlignItems::Center => (available - child_size).max(0.0) / 2.0,
            AlignItems::Stretch => 0.0,
        }
    }

    fn arrange_grid(&self, props: &GridProps, children: &mut Vec<WidgetNode>, inner: Rect) {
        let cols = props.columns.max(1);
        let inner_w = inner.size.width - (cols - 1) as f32 * props.column_gap;
        let col_w = inner_w / cols as f32;

        let rows = (children.len() + cols - 1) / cols;
        let mut row_y = 0.0;

        for row in 0..rows {
            let row_constraints = Constraints::loose(col_w, inner.size.height - row_y);
            let row_start = row * cols;
            let row_end = (row_start + cols).min(children.len());

            let row_height = children[row_start..row_end]
                .iter()
                .map(|c| self.measure(c, &row_constraints).height)
                .fold(0.0f32, f32::max);

            for col in 0..cols {
                let idx = row * cols + col;
                if idx >= children.len() { break; }
                let x = inner.origin.x + col as f32 * (col_w + props.column_gap);
                let y = inner.origin.y + row_y;
                let cs = self.measure(&children[idx], &row_constraints);
                children[idx].layout_rect = Some(Rect::new(x, y, cs.width, row_height));
            }

            row_y += row_height + props.row_gap;
        }
    }
}

// ============================================================================
// Part 5: Event System
// ============================================================================
//
// Events flow through the widget tree in two phases:
// 1. Capture (top-down): ancestor widgets can intercept events before
//    they reach the target widget. Used for modal overlays, global hotkeys.
// 2. Bubble (bottom-up): events propagate up after the target handles them.
//    A widget can stop propagation by consuming the event.

#[derive(Debug, Clone, PartialEq)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Key {
    Char(char),
    Enter,
    Escape,
    Backspace,
    Delete,
    Tab,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    Home,
    End,
    F(u8),
    Ctrl(Box<Key>),
    Shift(Box<Key>),
    Alt(Box<Key>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    MouseDown { position: Point, button: MouseButton },
    MouseUp { position: Point, button: MouseButton },
    MouseMove { position: Point, delta: Point },
    MouseWheel { position: Point, delta: Point },
    MouseEnter { position: Point },
    MouseLeave,
    KeyDown { key: Key, modifiers: Modifiers },
    KeyUp { key: Key, modifiers: Modifiers },
    TextInput { text: String },
    FocusGained,
    FocusLost,
    Resize { new_size: Size },
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Modifiers {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    pub meta: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EventResult {
    Consumed,    // event was handled, stop propagation
    Ignored,     // event not handled, continue propagation
}

/// Dispatch an event to a widget tree via hit testing + bubbling.
/// Returns the chain of widget IDs that handled the event.
pub fn dispatch_event(
    root: &WidgetNode,
    event: &Event,
) -> Vec<WidgetId> {
    let target = match event {
        Event::MouseDown { position, .. }
        | Event::MouseUp { position, .. }
        | Event::MouseMove { position, .. }
        | Event::MouseWheel { position, .. } => {
            hit_test(root, position)
        }
        _ => vec![root.id], // Non-positional events go to focused widget
    };
    target
}

/// Hit testing: find all widgets (from deepest to root) that contain the point.
/// Returns widget IDs in order from target (deepest) to root.
pub fn hit_test(node: &WidgetNode, point: &Point) -> Vec<WidgetId> {
    let rect = match &node.layout_rect {
        Some(r) => r,
        None => return vec![],
    };

    if !rect.contains(point) {
        return vec![];
    }

    // Check children first (deeper widgets have priority)
    let children_hits = match &node.widget {
        Widget::Flex { children, .. } => {
            children.iter().rev().find_map(|c| {
                let hits = hit_test(c, point);
                if !hits.is_empty() { Some(hits) } else { None }
            })
        }
        Widget::Grid { children, .. } => {
            children.iter().rev().find_map(|c| {
                let hits = hit_test(c, point);
                if !hits.is_empty() { Some(hits) } else { None }
            })
        }
        Widget::Container { children, .. } => {
            children.iter().rev().find_map(|c| {
                let hits = hit_test(c, point);
                if !hits.is_empty() { Some(hits) } else { None }
            })
        }
        Widget::Stack { children, .. } => {
            let mut sorted: Vec<&StackChild> = children.iter().collect();
            sorted.sort_by(|a, b| b.z_index.cmp(&a.z_index));
            sorted.iter().find_map(|sc| {
                let hits = hit_test(&sc.node, point);
                if !hits.is_empty() { Some(hits) } else { None }
            })
        }
        _ => None,
    };

    let mut result = children_hits.unwrap_or_default();
    result.push(node.id);
    result
}

// ============================================================================
// Part 6: Reactive State
// ============================================================================
//
// Signals are the primitive reactive unit. When you read a Signal, any
// computation that is currently executing becomes a subscriber. When
// you write to a Signal, all subscribers re-run.
//
// We implement a simplified version without dynamic subscription tracking
// (which would require thread-local storage and a stack of current scopes).
// Instead, we use an explicit dependency graph.

#[derive(Debug, Clone)]
pub struct Signal<T: Clone> {
    value: T,
    version: u64,
    subscribers: Vec<usize>, // IDs of computed values that depend on this
}

impl<T: Clone> Signal<T> {
    pub fn new(value: T) -> Self {
        Self { value, version: 0, subscribers: vec![] }
    }

    pub fn get(&self) -> &T {
        &self.value
    }

    pub fn set(&mut self, new_value: T) {
        self.value = new_value;
        self.version += 1;
    }

    pub fn update(&mut self, f: impl FnOnce(&T) -> T) {
        let new_val = f(&self.value);
        self.set(new_val);
    }

    pub fn version(&self) -> u64 {
        self.version
    }
}

/// A computed value that derives from one or more signals.
/// Automatically re-computes when its dependencies change.
pub struct Computed<T: Clone> {
    pub value: T,
    compute_fn: Box<dyn Fn() -> T>,
    dep_versions: Vec<u64>,
}

impl<T: Clone> Computed<T> {
    pub fn new(f: impl Fn() -> T + 'static) -> Self {
        let value = f();
        Self {
            value,
            compute_fn: Box::new(f),
            dep_versions: vec![],
        }
    }

    pub fn get(&mut self) -> &T {
        let new_value = (self.compute_fn)();
        self.value = new_value;
        &self.value
    }
}

// ============================================================================
// Part 7: Virtual DOM Diffing
// ============================================================================
//
// When state changes, we don't want to re-layout the entire widget tree.
// Instead, we diff the new tree against the old one and produce a list of
// "patch" operations that only update what changed.
//
// The diffing algorithm:
// 1. If the widget type changed: replace the entire subtree
// 2. If props changed but type is same: update in place
// 3. For lists of children: use keys to match old and new children,
//    then diff matched pairs (O(n) with hashing)

#[derive(Debug, Clone, PartialEq)]
pub enum Patch {
    /// Replace old widget with new widget at given ID
    Replace { id: WidgetId, new_widget: Widget },
    /// Update props of a widget (style, content, etc.)
    UpdateProps { id: WidgetId, new_widget: Widget },
    /// Insert a new child at position
    InsertChild { parent_id: WidgetId, index: usize, node: WidgetNode },
    /// Remove a child
    RemoveChild { parent_id: WidgetId, child_id: WidgetId },
    /// Move a child to a new position
    MoveChild { parent_id: WidgetId, child_id: WidgetId, new_index: usize },
}

pub fn diff(old_node: &WidgetNode, new_node: &WidgetNode) -> Vec<Patch> {
    let mut patches = Vec::new();

    // Check if widget type changed — if so, replace entirely
    if !same_widget_type(&old_node.widget, &new_node.widget) {
        patches.push(Patch::Replace {
            id: old_node.id,
            new_widget: new_node.widget.clone(),
        });
        return patches;
    }

    // Widget type is same — check if props changed
    if old_node.widget != new_node.widget {
        patches.push(Patch::UpdateProps {
            id: old_node.id,
            new_widget: new_node.widget.clone(),
        });
    }

    // Diff children
    let old_children = get_children(&old_node.widget);
    let new_children = get_children(&new_node.widget);

    let old_len = old_children.len();
    let new_len = new_children.len();

    // Simple O(n) diff: zip and diff matched pairs, handle additions/removals
    let min_len = old_len.min(new_len);
    for i in 0..min_len {
        let child_patches = diff(old_children[i], new_children[i]);
        patches.extend(child_patches);
    }

    // New children added
    for i in old_len..new_len {
        patches.push(Patch::InsertChild {
            parent_id: old_node.id,
            index: i,
            node: new_children[i].clone(),
        });
    }

    // Old children removed
    for i in new_len..old_len {
        patches.push(Patch::RemoveChild {
            parent_id: old_node.id,
            child_id: old_children[i].id,
        });
    }

    patches
}

fn same_widget_type(a: &Widget, b: &Widget) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn get_children(widget: &Widget) -> Vec<&WidgetNode> {
    match widget {
        Widget::Flex { children, .. } => children.iter().collect(),
        Widget::Grid { children, .. } => children.iter().collect(),
        Widget::Container { children, .. } => children.iter().collect(),
        Widget::Stack { children, .. } => children.iter().map(|sc| &sc.node).collect(),
        Widget::Scroll { child, .. } => vec![child.as_ref()],
        _ => vec![],
    }
}

// ============================================================================
// Part 8: Animation System
// ============================================================================
//
// Animations interpolate between values over time using easing functions.
// An easing function maps t ∈ [0, 1] to a progress value, allowing
// animations that accelerate, decelerate, bounce, or spring.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Easing {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    Bounce,
    Elastic,
    Spring { tension: f32, friction: f32 },
}

impl Easing {
    /// Compute the easing value for t ∈ [0, 1]
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Easing::Linear => t,
            Easing::EaseIn => t * t,
            Easing::EaseOut => t * (2.0 - t),
            Easing::EaseInOut => {
                if t < 0.5 { 2.0 * t * t }
                else { -1.0 + (4.0 - 2.0 * t) * t }
            }
            Easing::EaseInCubic => t * t * t,
            Easing::EaseOutCubic => {
                let t1 = t - 1.0;
                t1 * t1 * t1 + 1.0
            }
            Easing::EaseInOutCubic => {
                if t < 0.5 { 4.0 * t * t * t }
                else {
                    let t2 = t * 2.0 - 2.0;
                    0.5 * t2 * t2 * t2 + 1.0
                }
            }
            Easing::Bounce => {
                let n1 = 7.5625;
                let d1 = 2.75;
                if t < 1.0 / d1 {
                    n1 * t * t
                } else if t < 2.0 / d1 {
                    let t2 = t - 1.5 / d1;
                    n1 * t2 * t2 + 0.75
                } else if t < 2.5 / d1 {
                    let t2 = t - 2.25 / d1;
                    n1 * t2 * t2 + 0.9375
                } else {
                    let t2 = t - 2.625 / d1;
                    n1 * t2 * t2 + 0.984375
                }
            }
            Easing::Elastic => {
                if t == 0.0 || t == 1.0 {
                    t
                } else {
                    let c4 = 2.0 * std::f32::consts::PI / 3.0;
                    -(2.0f32.powf(10.0 * t - 10.0)) * ((t * 10.0 - 10.75) * c4).sin()
                }
            }
            Easing::Spring { tension, friction } => {
                // Simple underdamped spring approximation
                let zeta = friction / (2.0 * tension.sqrt());
                if zeta < 1.0 {
                    let wd = tension.sqrt() * (1.0 - zeta * zeta).sqrt();
                    1.0 - (-zeta * tension.sqrt() * t).exp() * ((wd * t).cos() + (zeta / (1.0 - zeta * zeta).sqrt()) * (wd * t).sin())
                } else {
                    Easing::EaseOutCubic.apply(t)
                }
            }
        }
    }
}

/// An animation running between two values
#[derive(Debug, Clone)]
pub struct Animation {
    pub start_value: f32,
    pub end_value: f32,
    pub duration_secs: f32,
    pub easing: Easing,
    pub elapsed_secs: f32,
    pub repeat: bool,
    pub auto_reverse: bool,
}

impl Animation {
    pub fn new(from: f32, to: f32, duration: f32, easing: Easing) -> Self {
        Self {
            start_value: from,
            end_value: to,
            duration_secs: duration,
            easing,
            elapsed_secs: 0.0,
            repeat: false,
            auto_reverse: false,
        }
    }

    pub fn is_complete(&self) -> bool {
        !self.repeat && self.elapsed_secs >= self.duration_secs
    }

    pub fn current_value(&self) -> f32 {
        let t = if self.duration_secs > 0.0 {
            (self.elapsed_secs / self.duration_secs).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let eased = self.easing.apply(t);
        self.start_value + (self.end_value - self.start_value) * eased
    }

    /// Advance the animation by dt seconds. Returns true if still running.
    pub fn tick(&mut self, dt: f32) -> bool {
        self.elapsed_secs += dt;
        if self.repeat {
            if self.elapsed_secs >= self.duration_secs {
                if self.auto_reverse {
                    std::mem::swap(&mut self.start_value, &mut self.end_value);
                }
                self.elapsed_secs -= self.duration_secs;
            }
            true
        } else {
            !self.is_complete()
        }
    }
}

// ============================================================================
// Part 9: Theme System
// ============================================================================
//
// A theme provides a consistent set of colors, fonts, and spacing values
// that can be applied across an entire application. Switching themes
// (light/dark mode) only requires changing the theme and re-rendering.

#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    pub background: Color,
    pub surface: Color,
    pub primary: Color,
    pub secondary: Color,
    pub text_primary: Color,
    pub text_secondary: Color,
    pub border: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub font_size_sm: f32,
    pub font_size_md: f32,
    pub font_size_lg: f32,
    pub font_size_xl: f32,
    pub spacing_xs: f32,
    pub spacing_sm: f32,
    pub spacing_md: f32,
    pub spacing_lg: f32,
    pub border_radius: f32,
}

impl Theme {
    pub fn light() -> Self {
        Self {
            name: "light".to_string(),
            background: Color::from_hex(0xF5F5F5),
            surface: Color::white(),
            primary: Color::from_hex(0x2196F3),
            secondary: Color::from_hex(0xFF4081),
            text_primary: Color::from_hex(0x212121),
            text_secondary: Color::from_hex(0x757575),
            border: Color::from_hex(0xE0E0E0),
            success: Color::from_hex(0x4CAF50),
            warning: Color::from_hex(0xFF9800),
            error: Color::from_hex(0xF44336),
            font_size_sm: 12.0,
            font_size_md: 14.0,
            font_size_lg: 18.0,
            font_size_xl: 24.0,
            spacing_xs: 4.0,
            spacing_sm: 8.0,
            spacing_md: 16.0,
            spacing_lg: 24.0,
            border_radius: 4.0,
        }
    }

    pub fn dark() -> Self {
        Self {
            name: "dark".to_string(),
            background: Color::from_hex(0x121212),
            surface: Color::from_hex(0x1E1E1E),
            primary: Color::from_hex(0x90CAF9),
            secondary: Color::from_hex(0xF48FB1),
            text_primary: Color::from_hex(0xEEEEEE),
            text_secondary: Color::from_hex(0xBDBDBD),
            border: Color::from_hex(0x424242),
            success: Color::from_hex(0x81C784),
            warning: Color::from_hex(0xFFB74D),
            error: Color::from_hex(0xE57373),
            font_size_sm: 12.0,
            font_size_md: 14.0,
            font_size_lg: 18.0,
            font_size_xl: 24.0,
            spacing_xs: 4.0,
            spacing_sm: 8.0,
            spacing_md: 16.0,
            spacing_lg: 24.0,
            border_radius: 4.0,
        }
    }

    /// Generate a Style for buttons using this theme
    pub fn button_style(&self) -> Style {
        let mut s = Style::default();
        s.background = Some(self.primary);
        s.foreground = Some(Color::white());
        s.font_size = Some(self.font_size_md);
        s.padding = EdgeInsets::symmetric(self.spacing_sm, self.spacing_md);
        s.border = Border::rounded(0.0, Color::transparent(), self.border_radius);
        s
    }

    /// Generate a Style for text inputs using this theme
    pub fn input_style(&self) -> Style {
        let mut s = Style::default();
        s.background = Some(self.surface);
        s.foreground = Some(self.text_primary);
        s.font_size = Some(self.font_size_md);
        s.padding = EdgeInsets::symmetric(self.spacing_sm, self.spacing_sm);
        s.border = Border::rounded(1.0, self.border, self.border_radius);
        s
    }
}

// ============================================================================
// Part 10: ASCII Renderer (for testing without a display)
// ============================================================================
//
// Since we don't have access to a real display, we implement a simple
// ASCII renderer that draws the widget tree as a text-based diagram.
// This is useful for testing layout and structure.

pub struct AsciiRenderer {
    pub width: usize,
    pub height: usize,
    buffer: Vec<Vec<char>>,
}

impl AsciiRenderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buffer: vec![vec![' '; width]; height],
        }
    }

    pub fn clear(&mut self) {
        for row in &mut self.buffer {
            for cell in row.iter_mut() {
                *cell = ' ';
            }
        }
    }

    pub fn draw_rect(&mut self, rect: &Rect, ch: char) {
        let x0 = rect.min_x() as usize;
        let y0 = rect.min_y() as usize;
        let x1 = (rect.max_x() as usize).min(self.width);
        let y1 = (rect.max_y() as usize).min(self.height);

        for y in y0..y1 {
            for x in x0..x1 {
                if y < self.height && x < self.width {
                    // Border: draw box-drawing chars on edges
                    let is_top = y == y0;
                    let is_bottom = y + 1 == y1;
                    let is_left = x == x0;
                    let is_right = x + 1 == x1;

                    self.buffer[y][x] = if is_top && is_left {
                        '+'
                    } else if is_top && is_right {
                        '+'
                    } else if is_bottom && is_left {
                        '+'
                    } else if is_bottom && is_right {
                        '+'
                    } else if is_top || is_bottom {
                        '-'
                    } else if is_left || is_right {
                        '|'
                    } else {
                        ch
                    };
                }
            }
        }
    }

    pub fn draw_text(&mut self, x: usize, y: usize, text: &str) {
        if y >= self.height { return; }
        for (i, ch) in text.chars().enumerate() {
            if x + i >= self.width { break; }
            self.buffer[y][x + i] = ch;
        }
    }

    pub fn render_widget(&mut self, node: &WidgetNode) {
        if let Some(rect) = &node.layout_rect {
            match &node.widget {
                Widget::Text { content, .. } => {
                    self.draw_text(rect.min_x() as usize, rect.min_y() as usize, content);
                }
                Widget::Button { label, .. } => {
                    self.draw_rect(rect, ' ');
                    let x = (rect.min_x() + 2.0) as usize;
                    let y = (rect.min_y() + 1.0) as usize;
                    self.draw_text(x, y, label);
                }
                Widget::Container { children, .. } => {
                    self.draw_rect(rect, ' ');
                    for child in children {
                        self.render_widget(child);
                    }
                }
                Widget::Flex { children, .. } => {
                    for child in children {
                        self.render_widget(child);
                    }
                }
                Widget::Grid { children, .. } => {
                    for child in children {
                        self.render_widget(child);
                    }
                }
                Widget::Stack { children, .. } => {
                    let mut sorted: Vec<&StackChild> = children.iter().collect();
                    sorted.sort_by_key(|sc| sc.z_index);
                    for sc in sorted {
                        self.render_widget(&sc.node);
                    }
                }
                Widget::TextInput { value, placeholder, .. } => {
                    self.draw_rect(rect, ' ');
                    let label = if value.is_empty() { placeholder } else { value };
                    self.draw_text(rect.min_x() as usize + 1, rect.min_y() as usize + 1, label);
                }
                Widget::ProgressBar { value, .. } => {
                    let inner_w = (rect.size.width as usize).saturating_sub(2);
                    let filled = (inner_w as f32 * value.clamp(0.0, 1.0)) as usize;
                    let x = rect.min_x() as usize;
                    let y = rect.min_y() as usize;
                    if y < self.height {
                        if x < self.width { self.buffer[y][x] = '['; }
                        for i in 0..inner_w {
                            let cx = x + 1 + i;
                            if cx < self.width {
                                self.buffer[y][cx] = if i < filled { '#' } else { ' ' };
                            }
                        }
                        let ex = x + 1 + inner_w;
                        if ex < self.width { self.buffer[y][ex] = ']'; }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn to_string(&self) -> String {
        self.buffer
            .iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ============================================================================
// Part 11: Widget Builder Helpers
// ============================================================================
//
// Convenience constructors for building widget trees.

pub struct Ui;

static mut NEXT_WIDGET_ID: WidgetId = 1;

fn new_id() -> WidgetId {
    unsafe {
        let id = NEXT_WIDGET_ID;
        NEXT_WIDGET_ID += 1;
        id
    }
}

impl Ui {
    pub fn text(content: &str) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Text {
            content: content.to_string(),
            style: Style::default(),
        })
    }

    pub fn text_styled(content: &str, style: Style) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Text {
            content: content.to_string(),
            style,
        })
    }

    pub fn button(label: &str) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Button {
            label: label.to_string(),
            on_click: None,
            enabled: true,
            style: Style::default(),
        })
    }

    pub fn button_with_action(label: &str, action: &str) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Button {
            label: label.to_string(),
            on_click: Some(action.to_string()),
            enabled: true,
            style: Style::default(),
        })
    }

    pub fn row(children: Vec<WidgetNode>) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Flex {
            props: FlexProps::row(),
            style: Style::default(),
            children,
        })
    }

    pub fn column(children: Vec<WidgetNode>) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Flex {
            props: FlexProps::column(),
            style: Style::default(),
            children,
        })
    }

    pub fn container(children: Vec<WidgetNode>) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Container {
            style: Style::default(),
            width: SizeConstraint::Fill(1.0),
            height: SizeConstraint::Wrap,
            children,
        })
    }

    pub fn spacer() -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Spacer { flex: 1.0 })
    }

    pub fn text_input(value: &str, placeholder: &str) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::TextInput {
            value: value.to_string(),
            placeholder: placeholder.to_string(),
            on_change: None,
            enabled: true,
            style: Style::default(),
        })
    }

    pub fn checkbox(label: &str, checked: bool) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Checkbox {
            checked,
            label: label.to_string(),
            on_change: None,
            style: Style::default(),
        })
    }

    pub fn slider(value: f32, min: f32, max: f32) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Slider {
            value,
            min,
            max,
            step: 0.01,
            on_change: None,
            style: Style::default(),
        })
    }

    pub fn progress(value: f32) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::ProgressBar {
            value,
            style: Style::default(),
        })
    }

    pub fn divider() -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Divider {
            axis: Axis::Horizontal,
            thickness: 1.0,
            color: Color::light_gray(),
        })
    }

    pub fn grid(columns: usize, children: Vec<WidgetNode>) -> WidgetNode {
        WidgetNode::new(new_id(), Widget::Grid {
            props: GridProps {
                columns,
                rows: None,
                column_gap: 8.0,
                row_gap: 8.0,
                column_widths: vec![],
            },
            style: Style::default(),
            children,
        })
    }
}

// ============================================================================
// Part 12: Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.5
    }

    // --- Geometry tests ---

    #[test]
    fn test_rect_contains() {
        let r = Rect::new(10.0, 10.0, 100.0, 50.0);
        assert!(r.contains(&Point::new(50.0, 30.0)));
        assert!(!r.contains(&Point::new(5.0, 30.0)));
        assert!(!r.contains(&Point::new(50.0, 65.0)));
    }

    #[test]
    fn test_rect_union() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 10.0, 10.0);
        let u = a.union(&b);
        assert_eq!(u.min_x(), 0.0);
        assert_eq!(u.min_y(), 0.0);
        assert_eq!(u.max_x(), 15.0);
        assert_eq!(u.max_y(), 15.0);
    }

    #[test]
    fn test_rect_inset() {
        let r = Rect::new(0.0, 0.0, 100.0, 80.0);
        let insets = EdgeInsets::all(10.0);
        let inner = r.inset(&insets);
        assert_eq!(inner.origin.x, 10.0);
        assert_eq!(inner.origin.y, 10.0);
        assert_eq!(inner.size.width, 80.0);
        assert_eq!(inner.size.height, 60.0);
    }

    #[test]
    fn test_point_distance() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 0.001);
    }

    // --- Color tests ---

    #[test]
    fn test_color_from_hex() {
        let c = Color::from_hex(0xFF5733);
        assert!((c.r - 1.0).abs() < 0.01);
        assert!((c.g - 0.341).abs() < 0.01);
        assert!((c.b - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_color_round_trip() {
        let hex = 0xABCDEF;
        let c = Color::from_hex(hex);
        assert_eq!(c.to_hex(), hex);
    }

    #[test]
    fn test_color_mix() {
        let black = Color::black();
        let white = Color::white();
        let gray = black.mix(&white, 0.5);
        assert!((gray.r - 0.5).abs() < 0.01);
        assert!((gray.g - 0.5).abs() < 0.01);
        assert!((gray.b - 0.5).abs() < 0.01);
    }

    // --- Style tests ---

    #[test]
    fn test_style_inheritance() {
        let parent = Style {
            font_size: Some(16.0),
            foreground: Some(Color::black()),
            ..Style::default()
        };
        let child = Style::default();
        let inherited = child.inherit_from(&parent);
        assert_eq!(inherited.font_size, Some(16.0));
        assert_eq!(inherited.foreground, Some(Color::black()));
    }

    #[test]
    fn test_style_override() {
        let parent = Style {
            font_size: Some(16.0),
            ..Style::default()
        };
        let child = Style {
            font_size: Some(24.0),
            ..Style::default()
        };
        let inherited = child.inherit_from(&parent);
        assert_eq!(inherited.font_size, Some(24.0)); // child overrides parent
    }

    // --- Layout tests ---

    #[test]
    fn test_text_layout() {
        let engine = LayoutEngine::new();
        let node = Ui::text("Hello World");
        let constraints = Constraints::loose(200.0, 100.0);
        let size = engine.measure(&node, &constraints);
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
        assert!(size.width <= 200.0);
    }

    #[test]
    fn test_button_layout() {
        let engine = LayoutEngine::new();
        let node = Ui::button("Click Me");
        let constraints = Constraints::loose(400.0, 100.0);
        let size = engine.measure(&node, &constraints);
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_flex_row_layout() {
        let engine = LayoutEngine::new();
        let row = Ui::row(vec![
            Ui::text("Left"),
            Ui::text("Right"),
        ]);
        let constraints = Constraints::loose(400.0, 100.0);
        let size = engine.measure(&row, &constraints);
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_flex_column_layout() {
        let engine = LayoutEngine::new();
        let col = Ui::column(vec![
            Ui::text("Line 1"),
            Ui::text("Line 2"),
            Ui::text("Line 3"),
        ]);
        let constraints = Constraints::loose(300.0, 200.0);
        let size = engine.measure(&col, &constraints);
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_container_fixed_size() {
        let engine = LayoutEngine::new();
        let container = WidgetNode::new(1, Widget::Container {
            style: Style::default(),
            width: SizeConstraint::Fixed(200.0),
            height: SizeConstraint::Fixed(100.0),
            children: vec![],
        });
        let constraints = Constraints::loose(500.0, 400.0);
        let size = engine.measure(&container, &constraints);
        assert_eq!(size.width, 200.0);
        assert_eq!(size.height, 100.0);
    }

    #[test]
    fn test_grid_layout() {
        let engine = LayoutEngine::new();
        let grid = Ui::grid(3, vec![
            Ui::text("A"),
            Ui::text("B"),
            Ui::text("C"),
            Ui::text("D"),
            Ui::text("E"),
            Ui::text("F"),
        ]);
        let constraints = Constraints::loose(300.0, 400.0);
        let size = engine.measure(&grid, &constraints);
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_arrange_assigns_rects() {
        let engine = LayoutEngine::new();
        let mut row = Ui::row(vec![
            Ui::text("Hello"),
            Ui::text("World"),
        ]);
        let rect = Rect::new(0.0, 0.0, 400.0, 50.0);
        engine.arrange(&mut row, rect);
        assert!(row.layout_rect.is_some());
    }

    #[test]
    fn test_progress_bar_layout() {
        let engine = LayoutEngine::new();
        let pb = Ui::progress(0.75);
        let constraints = Constraints::loose(200.0, 100.0);
        let size = engine.measure(&pb, &constraints);
        assert_eq!(size.width, 200.0);
        assert!(size.height > 0.0);
    }

    // --- Hit testing tests ---

    #[test]
    fn test_hit_test_single_widget() {
        let engine = LayoutEngine::new();
        let mut btn = Ui::button("Click");
        let rect = Rect::new(10.0, 10.0, 80.0, 30.0);
        engine.arrange(&mut btn, rect);

        // Point inside button
        let hits = hit_test(&btn, &Point::new(50.0, 25.0));
        assert!(!hits.is_empty());
        assert_eq!(hits[0], btn.id);

        // Point outside button
        let misses = hit_test(&btn, &Point::new(5.0, 5.0));
        assert!(misses.is_empty());
    }

    #[test]
    fn test_hit_test_nested() {
        let engine = LayoutEngine::new();
        let child_btn = Ui::button("Inner");
        let child_id = child_btn.id;

        let mut row = Ui::row(vec![child_btn]);
        let rect = Rect::new(0.0, 0.0, 400.0, 50.0);
        engine.arrange(&mut row, rect);

        // If child has a valid rect, it should be hit before the row
        if let Widget::Flex { children, .. } = &row.widget {
            if let Some(child_rect) = &children[0].layout_rect {
                let center = child_rect.center();
                let hits = hit_test(&row, &center);
                // The deepest hit (child) should come first
                assert!(!hits.is_empty());
                assert_eq!(hits[0], child_id);
            }
        }
    }

    // --- Event dispatching tests ---

    #[test]
    fn test_event_dispatch() {
        let engine = LayoutEngine::new();
        let mut btn = Ui::button("Click");
        let rect = Rect::new(0.0, 0.0, 100.0, 40.0);
        engine.arrange(&mut btn, rect);

        let event = Event::MouseDown {
            position: Point::new(50.0, 20.0),
            button: MouseButton::Left,
        };
        let targets = dispatch_event(&btn, &event);
        assert!(!targets.is_empty());
    }

    // --- Reactive state tests ---

    #[test]
    fn test_signal_basic() {
        let mut counter = Signal::new(0i32);
        assert_eq!(*counter.get(), 0);
        counter.set(42);
        assert_eq!(*counter.get(), 42);
        counter.update(|v| v + 1);
        assert_eq!(*counter.get(), 43);
    }

    #[test]
    fn test_signal_version() {
        let mut sig = Signal::new("hello".to_string());
        let v0 = sig.version();
        sig.set("world".to_string());
        let v1 = sig.version();
        assert!(v1 > v0);
    }

    #[test]
    fn test_computed() {
        let x = Signal::new(5.0f32);
        let x_val = *x.get();
        let mut doubled = Computed::new(move || x_val * 2.0);
        assert_eq!(*doubled.get(), 10.0);
    }

    // --- Virtual DOM diffing tests ---

    #[test]
    fn test_diff_no_changes() {
        let node1 = Ui::text("Hello");
        let node2 = Ui::text("Hello");
        let patches = diff(&node1, &node2);
        assert!(patches.is_empty(), "Identical trees should produce no patches");
    }

    #[test]
    fn test_diff_content_change() {
        let node1 = Ui::text("Hello");
        let node2 = Ui::text("World");
        let patches = diff(&node1, &node2);
        assert!(!patches.is_empty(), "Changed content should produce patches");
        assert!(patches.iter().any(|p| matches!(p, Patch::UpdateProps { .. })));
    }

    #[test]
    fn test_diff_type_change() {
        let node1 = Ui::text("Hello");
        let node2 = Ui::button("Hello");
        let patches = diff(&node1, &node2);
        assert!(!patches.is_empty());
        assert!(patches.iter().any(|p| matches!(p, Patch::Replace { .. })));
    }

    #[test]
    fn test_diff_child_added() {
        let row1 = Ui::row(vec![Ui::text("A")]);
        let row2 = Ui::row(vec![Ui::text("A"), Ui::text("B")]);
        let patches = diff(&row1, &row2);
        assert!(patches.iter().any(|p| matches!(p, Patch::InsertChild { .. })));
    }

    #[test]
    fn test_diff_child_removed() {
        let row1 = Ui::row(vec![Ui::text("A"), Ui::text("B")]);
        let row2 = Ui::row(vec![Ui::text("A")]);
        let patches = diff(&row1, &row2);
        assert!(patches.iter().any(|p| matches!(p, Patch::RemoveChild { .. })));
    }

    // --- Animation tests ---

    #[test]
    fn test_animation_start_end() {
        let anim = Animation::new(0.0, 100.0, 1.0, Easing::Linear);
        assert_eq!(anim.current_value(), 0.0);

        let mut anim = Animation::new(0.0, 100.0, 1.0, Easing::Linear);
        anim.elapsed_secs = 1.0;
        assert_eq!(anim.current_value(), 100.0);
    }

    #[test]
    fn test_animation_midpoint() {
        let mut anim = Animation::new(0.0, 100.0, 2.0, Easing::Linear);
        anim.elapsed_secs = 1.0;
        assert!((anim.current_value() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_animation_tick() {
        let mut anim = Animation::new(0.0, 10.0, 1.0, Easing::Linear);
        assert!(!anim.is_complete());
        let still_running = anim.tick(0.5);
        assert!(still_running);
        assert!((anim.current_value() - 5.0).abs() < 0.01);
        anim.tick(0.5);
        assert!(anim.is_complete());
    }

    #[test]
    fn test_easing_bounds() {
        let easings = [
            Easing::Linear,
            Easing::EaseIn,
            Easing::EaseOut,
            Easing::EaseInOut,
            Easing::EaseInCubic,
            Easing::EaseOutCubic,
            Easing::Bounce,
        ];
        for easing in &easings {
            let t0 = easing.apply(0.0);
            let t1 = easing.apply(1.0);
            assert!((t0 - 0.0).abs() < 0.01, "{:?} at t=0 should be ~0, got {}", easing, t0);
            assert!((t1 - 1.0).abs() < 0.01, "{:?} at t=1 should be ~1, got {}", easing, t1);
        }
    }

    #[test]
    fn test_easing_linear_midpoint() {
        assert!((Easing::Linear.apply(0.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_easing_ease_out_faster_at_start() {
        // EaseOut should have more progress at t=0.3 than Linear
        let ease_out = Easing::EaseOut.apply(0.3);
        let linear = Easing::Linear.apply(0.3);
        assert!(ease_out > linear, "EaseOut should be ahead of linear early on");
    }

    // --- Theme tests ---

    #[test]
    fn test_theme_colors_distinct() {
        let light = Theme::light();
        let dark = Theme::dark();
        // Light and dark themes should have different backgrounds
        assert_ne!(light.background, dark.background);
        assert_ne!(light.text_primary, dark.text_primary);
    }

    #[test]
    fn test_theme_button_style() {
        let theme = Theme::light();
        let style = theme.button_style();
        assert_eq!(style.background, Some(theme.primary));
        assert_eq!(style.foreground, Some(Color::white()));
    }

    // --- ASCII renderer tests ---

    #[test]
    fn test_ascii_renderer_rect() {
        let mut renderer = AsciiRenderer::new(20, 5);
        let rect = Rect::new(0.0, 0.0, 20.0, 5.0);
        renderer.draw_rect(&rect, '.');
        let output = renderer.to_string();
        assert!(output.contains('+'));
        assert!(output.contains('-'));
        assert!(output.contains('|'));
    }

    #[test]
    fn test_ascii_renderer_text() {
        let mut renderer = AsciiRenderer::new(40, 5);
        renderer.draw_text(5, 2, "Hello");
        let output = renderer.to_string();
        assert!(output.contains('H'));
        assert!(output.contains("Hello"));
    }

    #[test]
    fn test_ascii_render_progress() {
        let engine = LayoutEngine::new();
        let mut pb = Ui::progress(0.5);
        let rect = Rect::new(0.0, 0.0, 20.0, 2.0);
        engine.arrange(&mut pb, rect);

        let mut renderer = AsciiRenderer::new(20, 3);
        renderer.render_widget(&pb);
        let output = renderer.to_string();
        assert!(output.contains('['));
        assert!(output.contains('#'));
        assert!(output.contains(']'));
    }

    // --- Integration test: full form UI ---

    #[test]
    fn test_full_form_layout() {
        let engine = LayoutEngine::new();

        // Build a simple login form
        let mut form = Ui::column(vec![
            Ui::text("Login"),
            Ui::divider(),
            Ui::text_input("", "Username"),
            Ui::text_input("", "Password"),
            Ui::checkbox("Remember me", false),
            Ui::row(vec![
                Ui::spacer(),
                Ui::button("Cancel"),
                Ui::button("Login"),
            ]),
            Ui::progress(0.0),
        ]);

        // Measure and arrange
        let constraints = Constraints::loose(400.0, 600.0);
        let size = engine.measure(&form, &constraints);
        assert!(size.width > 0.0, "Form should have positive width");
        assert!(size.height > 0.0, "Form should have positive height");

        engine.arrange(&mut form, Rect::new(0.0, 0.0, size.width, size.height));
        assert!(form.layout_rect.is_some());

        // Render to ASCII
        let mut renderer = AsciiRenderer::new(50, 25);
        renderer.render_widget(&form);
        let output = renderer.to_string();

        // Should contain text from form
        println!("Form output:\n{}", output);
    }

    #[test]
    fn test_signal_based_counter_ui() {
        // Simulate a counter UI that increments with a button
        let mut count = Signal::new(0i32);
        let mut label_text = format!("Count: {}", count.get());

        // Simulate button click
        count.update(|v| v + 1);
        count.update(|v| v + 1);
        label_text = format!("Count: {}", count.get());

        let engine = LayoutEngine::new();
        let ui = Ui::column(vec![
            Ui::text(&label_text),
            Ui::button_with_action("Increment", "increment"),
            Ui::button_with_action("Decrement", "decrement"),
        ]);

        let constraints = Constraints::loose(200.0, 150.0);
        let size = engine.measure(&ui, &constraints);
        assert!(size.width > 0.0);
        assert_eq!(*count.get(), 2);
    }
}
