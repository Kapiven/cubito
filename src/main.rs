use image::GenericImageView;
use nalgebra_glm::{Vec3, normalize};
use minifb::{Key, Window, WindowOptions};
use std::time::Duration;
use std::f32::consts::PI;

use rayon::prelude::*;

mod framebuffer;
mod ray_intersect;
mod cube; 
mod color;
mod camera;
mod light;
mod material;

use framebuffer::Framebuffer;
use cube::Cube;
use color::Color;
use ray_intersect::{Intersect, RayIntersect};
use camera::Camera;
use light::Light;
use material::Material;

const SHADOW_BIAS: f32 = 1e-4;
const MAX_RAY_DEPTH: u32 = 1; 

fn reflect(incident: &Vec3, normal: &Vec3) -> Vec3 {
    incident - 2.0 * incident.dot(normal) * normal
}

fn cast_shadow(
    intersect: &Intersect,
    light: &Light,
    objects: &[Cube],
) -> f32 {
    let light_dir = (light.position - intersect.point).normalize();
    let light_distance = (light.position - intersect.point).magnitude();

    let offset_normal = intersect.normal * SHADOW_BIAS;
    let shadow_ray_origin = if light_dir.dot(&intersect.normal) < 0.0 {
        intersect.point - offset_normal
    } else {
        intersect.point + offset_normal
    };

    let mut shadow_intensity = 0.0;
    for object in objects {
        let shadow_intersect = object.ray_intersect(&shadow_ray_origin, &light_dir);
        if shadow_intersect.is_intersecting && shadow_intersect.distance > 1e-3 && shadow_intersect.distance < light_distance {
            let distance_ratio = shadow_intersect.distance / light_distance;
            shadow_intensity = 1.0 - distance_ratio.powf(2.0).min(1.0);
            break;
        }
    }
    shadow_intensity
}

pub fn cast_ray(
    ray_origin: &Vec3,
    ray_direction: &Vec3,
    objects: &[Cube],
    lights: &[Light],
    depth: u32,
) -> Color {
    if depth > MAX_RAY_DEPTH {
        return Color::new(135.0, 206.0, 235.0); // sky blue
    }

    let mut intersect = Intersect::empty();
    let mut zbuffer = f32::INFINITY;

    for object in objects {
        let i = object.ray_intersect(ray_origin, ray_direction);
        if i.is_intersecting && i.distance < zbuffer {
            zbuffer = i.distance;
            intersect = i;
        }
    }

    if !intersect.is_intersecting {
        return Color::new(135.0, 206.0, 235.0);
    }

    let view_dir = (ray_origin - intersect.point).normalize();
    let mut result_color = Color::new(0.0, 0.0, 0.0);

    let is_crystal = intersect.material.is_crystal;

    // Color base: textura si existe, sino color difuso
    let mut base_color = intersect.material.diffuse;
    if let Some(tex) = &intersect.material.texture {
        if let Some((u, v)) = intersect.uv {
            let (tw, th) = tex.dimensions();
            let tx = ((u.clamp(0.0, 1.0)) * (tw - 1) as f32) as u32;
            let ty = ((v.clamp(0.0, 1.0)) * (th - 1) as f32) as u32;
            let pixel = tex.get_pixel(tx, ty);
            base_color = Color::new(
                pixel[0] as f32,
                pixel[1] as f32,
                pixel[2] as f32,
            );
        }
    }

    let ambient = base_color * 0.3;
    let mut lighting_color = ambient;

    for light in lights {
        let light_dir = (light.position - intersect.point).normalize();
        let reflect_dir = reflect(&-light_dir, &intersect.normal);

        let shadow_intensity = cast_shadow(&intersect, light, objects);
        let lit_amount = 1.0 - shadow_intensity;

        let diffuse_intensity = intersect.normal.dot(&light_dir).max(0.0).min(1.0);
        let diffuse = base_color * intersect.material.albedo[0] * diffuse_intensity * light.intensity * lit_amount;

        let specular_intensity = view_dir.dot(&reflect_dir).max(0.0).powf(intersect.material.specular);
        let specular = light.color * intersect.material.albedo[1] * specular_intensity * light.intensity * lit_amount;

        lighting_color = lighting_color + diffuse + specular;
    }
    result_color = lighting_color;

    if is_crystal {
        let reflect_dir = reflect(ray_direction, &intersect.normal).normalize();
        let reflect_origin = if reflect_dir.dot(&intersect.normal) < 0.0 {
            intersect.point - intersect.normal * SHADOW_BIAS
        } else {
            intersect.point + intersect.normal * SHADOW_BIAS
        };
        let reflect_color = cast_ray(&reflect_origin, &reflect_dir, objects, lights, depth + 1);

        let ior = 1.5;
        let mut refract_dir = ray_direction.clone();
        let mut refract_color = Color::new(0.0, 0.0, 0.0);
        let cosi = (-ray_direction).dot(&intersect.normal).max(-1.0).min(1.0);
        let etai = 1.0;
        let etat = ior;
        let n = if cosi < 0.0 { -intersect.normal } else { intersect.normal };
        let eta = if cosi < 0.0 { etat / etai } else { etai / etat };
        let k = 1.0 - eta * eta * (1.0 - cosi * cosi);
        if k >= 0.0 {
            refract_dir = (ray_direction * eta + n * (eta * cosi - k.sqrt())).normalize();
            let refract_origin = if refract_dir.dot(&intersect.normal) < 0.0 {
                intersect.point - intersect.normal * SHADOW_BIAS
            } else {
                intersect.point + intersect.normal * SHADOW_BIAS
            };
            refract_color = cast_ray(&refract_origin, &refract_dir, objects, lights, depth + 1);
        }

        let reflectance = 0.5;
        return reflect_color * reflectance + refract_color * (1.0 - reflectance);
    }

    result_color
}

// Render con Rayon
pub fn render(framebuffer: &mut Framebuffer, objects: &[Cube], camera: &Camera, lights: &[Light]) {
    let width = framebuffer.width as f32;
    let height = framebuffer.height as f32;
    let aspect_ratio = width / height;
    let fov = PI/3.0;
    let perspective_scale = (fov * 0.5).tan();

    // Paralelizamos por filas
    framebuffer.buffer
        .par_chunks_mut(framebuffer.width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..framebuffer.width {
                let screen_x = (2.0 * x as f32) / width - 1.0;
                let screen_y = -(2.0 * y as f32) / height + 1.0;

                let screen_x = screen_x * aspect_ratio * perspective_scale;
                let screen_y = screen_y * perspective_scale;

                let ray_direction = normalize(&Vec3::new(screen_x, screen_y, -1.0));
                let rotated_direction = camera.basis_change(&ray_direction);
                let pixel_color = cast_ray(&camera.position, &rotated_direction, objects, lights, 0);
                row[x] = pixel_color.to_hex();
            }
        });
}

fn main() {
    let window_width = 800;
    let window_height = 600;
    let framebuffer_width = 400;  
    let framebuffer_height = 300; 
    let frame_delay = Duration::from_millis(16);

    let mut framebuffer = Framebuffer::new(framebuffer_width, framebuffer_height);
    let mut window = Window::new(
        "Cubito",
        window_width,
        window_height,
        WindowOptions::default(),
    ).unwrap();

    window.set_position(500, 500);
    window.update();

    // Material texturizado
    let textured_cube = Material::with_texture(
        "./assets/flores.webp",
        80.0,
        [0.7, 0.3],
    );

    let light1 = Light::new(Vec3::new(0.0, 0.0, 5.0), Color::new(255.0, 200.0, 100.0), 1.0);
    let light2 = Light::new(Vec3::new(3.0, 4.0, 6.0), Color::new(100.0, 200.0, 255.0), 0.8);
    let lights = [light1, light2];

    let objects = [
        Cube { center: Vec3::new(0.0, 0.0, 0.0), size: 1.5, material: textured_cube },
    ];

    let mut camera = Camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0)
    );

    let mut yaw_velocity: f32 = 0.0;
    let mut pitch_velocity: f32 = 0.0;
    let acceleration: f32 = PI / 200.0;
    let damping: f32 = 0.85;
    let max_velocity: f32 = PI / 30.0;

    while window.is_open() {
        if window.is_key_down(Key::Escape) { break; }

        if window.is_key_down(Key::A) { yaw_velocity = (yaw_velocity + acceleration).min(max_velocity); }
        if window.is_key_down(Key::D) { yaw_velocity = (yaw_velocity - acceleration).max(-max_velocity); }
        if window.is_key_down(Key::W) { pitch_velocity = (pitch_velocity - acceleration).max(-max_velocity); }
        if window.is_key_down(Key::S) { pitch_velocity = (pitch_velocity + acceleration).min(max_velocity); }

        camera.orbit(yaw_velocity, pitch_velocity);
        yaw_velocity *= damping;
        pitch_velocity *= damping;

        render(&mut framebuffer, &objects, &camera, &lights);

        window.update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height).unwrap();
        std::thread::sleep(frame_delay);
    }
}