#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub fn to_hex(&self) -> u32 {
        ((self.r.clamp(0.0, 255.0) as u32) << 16)
            | ((self.g.clamp(0.0, 255.0) as u32) << 8)
            | (self.b.clamp(0.0, 255.0) as u32)
    }

    pub fn black() -> Self {
        Color::new(0.0, 0.0, 0.0)
    }

    pub fn blend(self, other: Color, factor: f32) -> Color {
        let f = factor.clamp(0.0, 1.0);
        Color {
            r: (self.r * (1.0 - f) + other.r * f),
            g: (self.g * (1.0 - f) + other.g * f),
            b: (self.b * (1.0 - f) + other.b * f),
        }
    }
}

use std::ops::{Add, Mul};

impl Add for Color {
    type Output = Color;
    fn add(self, other: Color) -> Color {
        Color {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

impl Mul<f32> for Color {
    type Output = Color;

    fn mul(self, scalar: f32) -> Color {
        Color {
            r: self.r * scalar,
            g: self.g * scalar,
            b: self.b * scalar,
        }
    }
}
