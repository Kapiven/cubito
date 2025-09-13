use crate::color::Color;
use image::DynamicImage;

#[derive(Debug, Clone)]
pub struct Material {
    pub diffuse: Color,
    pub specular: f32,
    pub albedo: [f32; 2],
    pub texture: Option<DynamicImage>,
    pub is_crystal: bool,
}

impl Material {
    pub fn new(diffuse: Color, specular: f32, albedo: [f32; 2]) -> Self {
        Self {
            diffuse,
            specular,
            albedo,
            texture: None,
            is_crystal: false,
        }
    }

    pub fn with_texture(path: &str, specular: f32, albedo: [f32; 2]) -> Self {
        let img = image::open(path).expect("No se pudo cargar la textura");
        Self {
            diffuse: Color::new(255.0, 255.0, 255.0),
            specular,
            albedo,
            texture: Some(img),
            is_crystal: false,
        }
    }

    pub fn crystal(diffuse: Color, specular: f32, albedo: [f32; 2]) -> Self {
        Self {
            diffuse,
            specular,
            albedo,
            texture: None,
            is_crystal: true,
        }
    }

    pub fn black() -> Self {
        Self {
            diffuse: Color::new(0.0, 0.0, 0.0),
            specular: 0.0,
            albedo: [0.0, 0.0],
            texture: None,
            is_crystal: false,
        }
    }
}
