use nalgebra_glm::Vec3;
use crate::material::Material;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Intersect {
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub is_intersecting: bool,
    pub material: Material,
    pub uv: Option<(f32, f32)>,
}

impl Intersect {
    pub fn new(point: Vec3, normal: Vec3, distance: f32, material: Material, uv: Option<(f32, f32)>) -> Self {
        Intersect {
            point,
            normal,
            distance,
            is_intersecting: true,
            material,
            uv,
        }
    }

    pub fn empty() -> Self {
        Intersect {
            point: Vec3::zeros(),
            normal: Vec3::zeros(),
            distance: 0.0,
            is_intersecting: false,
            material: Material::black(),
            uv: None,
        }
    }
}

pub trait RayIntersect {
  fn ray_intersect(&self, ray_origin: &Vec3, ray_direction: &Vec3) -> Intersect;
}



