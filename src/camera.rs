use nalgebra_glm::Vec3;
use std::f32::consts::PI;

pub struct Camera {
    pub position: Vec3, // Camera position in world space
    pub center: Vec3,   // Point the camera is looking at
    pub up: Vec3,      // Up vector
}

impl Camera {
    pub fn new(position: Vec3, center: Vec3, up: Vec3) -> Self {
        Camera {
            position,
            center,
            up,
        }
    }

    pub fn basis_change(&self, vector: &Vec3) -> Vec3 {
        let forward = (self.center - self.position).normalize();
        let right = forward.cross(&self.up).normalize();
        let up = right.cross(&forward).normalize();

        let rotated = 
        vector.x * right +
        vector.y * up -
        vector.z * forward;

        rotated.normalize()
    }

    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        // Calculate the vector from the center to the eye (radius vector) and measure the distance
        let radius_vector = self.position - self.center;
        let radius = radius_vector.magnitude();

        // Calculate current yaw (rotation around Y-axis)
        // atan2(z, x) gives us the angle in the XZ plane
        // Range: [-π, π], where 0 is along positive X-axis, π/2 is along positive Z-axis
        let current_yaw = radius_vector.z.atan2(radius_vector.x);

        // Calculate current pitch (rotation around X-axis)
        // xz here refers to the proyection of the radius over the x axis
        let radius_xz = (radius_vector.x * radius_vector.x + radius_vector.z * radius_vector.z).sqrt();
        // We use -y because positive pitch is when we look up (negative y in our coordinate system)
        // Range: [-π/2, π/2], where 0 is horizontal, π/2 is looking straight up
        let current_pitch = (-radius_vector.y).atan2(radius_xz);

        // Apply delta rotations
        // Keep yaw in range [0, 2π] for consistency
        let new_yaw = (current_yaw + delta_yaw) % (2.0 * PI);
        // Clamp pitch to slightly less than [-π/2, π/2] to prevent gimbal lock
        let new_pitch = (current_pitch + delta_pitch).clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);

        // Calculate new eye position
        // We use spherical coordinates to cartesian conversion:
        // x = r * cos(yaw) * cos(pitch)
        // y = -r * sin(pitch)  // Negative because positive y is up
        // z = r * sin(yaw) * cos(pitch)

        let new_position = self.center + Vec3::new(
            radius * new_yaw.cos() * new_pitch.cos(),
            -radius * new_pitch.sin(),
            radius * new_yaw.sin() * new_pitch.cos()
        );

        self.position = new_position;
    }
}