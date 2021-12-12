using Images
using StaticArrays
using LinearAlgebra


squared_length(v::SVector{3,Float32}) = v ⋅ v
near_zero(v::SVector) = squared_length(v) < 1e-5


#= Manipulating images =#
rgb(v::SVector{3,Float32}) = RGB(v[1], v[2], v[3]) # IIRC RGB(v...) was much slower...
rgb_gamma2(v::SVector{3,Float32}) = RGB(sqrt.(v)...) # claforte: try to expand?

struct Ray
    origin::SVector{3,Float32}
    dir::SVector{3,Float32} # direction (unit vector)
end

# equivalent to C++'s ray.at()
"3D Point of ray `r` evaluated at parameter t"
point(r::Ray, t::Float32)::SVector{3,Float32} = r.origin .+ t .* r.dir


#= Random vectors =#

# equiv to random_double()
random_between(min=0.0f0, max=1.0f0) = rand(Float32)*(max-min) + min
random_vec3(min::Float32, max::Float32) = @SVector[random_between(min,max) for i∈1:3]
random_vec2(min::Float32, max::Float32) = @SVector[random_between(min,max) for i∈1:2]
function random_vec3_in_sphere() # equiv to random_in_unit_sphere()
    while true
        p = random_vec3(-1f0, 1f0)
        if p⋅p <= 1
            return p
        end
    end
end

"Random unit vector. Equivalent to C++'s `unit_vector(random_in_unit_sphere())`"
random_vec3_on_sphere() = normalize(random_vec3_in_sphere())
function random_vec2_in_disk() # equiv to random_in_unit_disk()
    while true
        p = random_vec2(-1f0, 1f0)
        if p⋅p <= 1
            return p
        end
    end
end


#= Rays, simple camera, and background =#

function skycolor(ray::Ray)
    # NOTE: unlike in the C++ implem., we assume the ray direction is pre-normalized.
    white = @SVector[1.0f0,1.0f0,1.0f0]
    skyblue = @SVector[0.5f0,0.7f0,1.0f0]
    t = 0.5f0(ray.dir[2] + 1.0f0)
    (1.0f0-t)*white + t*skyblue
end


"An object that can be hit by Ray"
abstract type Hittable end

"""Materials tell us how rays interact with a surface"""
abstract type Material end


"Record a hit between a ray and an object's surface"
mutable struct HitRecord
    t::Float32 # vector from the ray's origin to the intersection with a surface. 
    
    # If t==Inf32, there was no hit, and all following values are undefined!
    #
    p::SVector{3,Float32} # point of intersection between an object's surface and ray
    n⃗::SVector{3,Float32} # local normal (see diagram below)
    
    # If true, our ray hit from outside to the front of the surface. 
    # If false, the ray hit from within.
    front_face::Bool
    mat::Material

    HitRecord() = new(Inf32) # no hit!
    HitRecord(t,p,n⃗,front_face,mat) = new(t,p,n⃗,front_face,mat)
end

struct Sphere <: Hittable
    center::SVector{3,Float32}
    radius::Float32
    mat::Material
end


"""Equivalent to `hit_record.set_face_normal()`"""
function ray_to_HitRecord(t, p, outward_n⃗, r_dir::SVector{3,Float32}, mat::Material)
    front_face = r_dir ⋅ outward_n⃗ < 0
    n⃗ = front_face ? outward_n⃗ : -outward_n⃗
    rec = HitRecord(t,p,n⃗,front_face,mat)
end

struct Scatter
    r::Ray
    attenuation::SVector{3,Float32}
    
    # claforte: TODO: rename to "absorbed?", i.e. not reflected/refracted?
    reflected::Bool # whether the scattered ray was reflected, or fully absorbed
    Scatter(r,a,reflected=true) = new(r,a,reflected)
end

"Diffuse material"
mutable struct Lambertian<:Material
    albedo::SVector{3,Float32}
end

"""Compute reflection vector for v (pointing to surface) and normal n⃗.

    See [diagram](https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection)
"""
reflect(v::SVector{3,Float32}, n⃗::SVector{3,Float32}) = v - (2v⋅n⃗)*n⃗

"""Create a scattered ray emitted by `mat` from incident Ray `r`. 

    Args:
        rec: the HitRecord of the surface from which to scatter the ray.

    Return missing if it's fully absorbed.
"""
function scatter(mat::Lambertian, r::Ray, rec::HitRecord)::Scatter
    scatter_dir = rec.n⃗ + random_vec3_on_sphere()
    if near_zero(scatter_dir) # Catch degenerate scatter direction
        scatter_dir = rec.n⃗ 
    else
        scatter_dir = normalize(scatter_dir)
    end
    scattered_r = Ray(rec.p, scatter_dir)
    attenuation = mat.albedo
    return Scatter(scattered_r, attenuation)
end

const _no_hit = HitRecord() # will be reused 

function hit(s::Sphere, r::Ray, tmin::Float32, tmax::Float32)
    oc = r.origin - s.center
    a = 1 #r.dir ⋅ r.dir # normalized vector - always 1
    half_b = oc ⋅ r.dir
    c = oc⋅oc - s.radius^2
    discriminant = half_b^2 - a*c
    if discriminant < 0 return _no_hit end
    sqrtd = √discriminant
    
    # Find the nearest root that lies in the acceptable range
    root = (-half_b - sqrtd) / a    
    if root < tmin || tmax < root
        root = (-half_b + sqrtd) / a
        if root < tmin || tmax < root
            return _no_hit
        end
    end
        
    t = root
    p = point(r, t)
    n⃗ = (p - s.center) / s.radius
    return ray_to_HitRecord(t, p, n⃗, r.dir, s.mat)
end

struct HittableList <: Hittable
    list::Vector{Hittable}
end

"""Find closest hit between `Ray r` and a list of Hittable objects `h`, within distance `tmin` < `tmax`"""
function hit(hittables::HittableList, r::Ray, tmin::Float32, tmax::Float32)
    closest = tmax # closest t so far
    rec = _no_hit
    for h in hittables.list
        temprec = hit(h, r, tmin, closest)
        if temprec !== _no_hit
            rec = temprec
            closest = rec.t # i.e. ignore any further hit > this one's.
        end
    end
    rec
end


# Metal material

mutable struct Metal<:Material
    albedo::SVector{3,Float32}
    fuzz::Float32 # how big the sphere used to generate fuzzy reflection rays. 0=none
    Metal(a,f=0.0) = new(a,f)
end

function scatter(mat::Metal, r_in::Ray, rec::HitRecord)::Scatter
    reflected = normalize(reflect(r_in.dir, rec.n⃗) + mat.fuzz*random_vec3_on_sphere())
    Scatter(Ray(rec.p, reflected), mat.albedo)
end

mutable struct Camera
    origin::SVector{3,Float32}
    lower_left_corner::SVector{3,Float32}
    horizontal::SVector{3,Float32}
    vertical::SVector{3,Float32}
    u::SVector{3,Float32}
    v::SVector{3,Float32}
    w::SVector{3,Float32}
    lens_radius::Float32
end

"""
    Args:
        vfov: vertical field-of-view in degrees
        aspect_ratio: horizontal/vertical ratio of pixels
      aperture: if 0 - no depth-of-field
"""
function default_camera(lookfrom::SVector{3,Float32}=@SVector[0f0,0f0,0f0], 
            lookat::SVector{3,Float32}=@SVector[0f0,0f0,-1f0], 
            vup::SVector{3,Float32}=@SVector[0f0,1f0,0f0], vfov=90.0f0, aspect_ratio=16.0f0/9.0f0,
                        aperture=0.0f0, focus_dist=1.0f0)
    viewport_height = 2.0f0 * tand(vfov/2f0)
    viewport_width = aspect_ratio * viewport_height
    
    w = normalize(lookfrom - lookat)
    u = normalize(vup × w)
    v = w × u
    
    origin = lookfrom
    horizontal = focus_dist * viewport_width * u
    vertical = focus_dist * viewport_height * v
    lower_left_corner = origin - horizontal/2f0 - vertical/2f0 - focus_dist*w
    lens_radius = aperture/2f0
    Camera(origin, lower_left_corner, horizontal, vertical, u, v, w, lens_radius)
end

function get_ray(c::Camera, s::Float32, t::Float32)
    rd = SVector{2,Float32}(c.lens_radius * random_vec2_in_disk())
    offset = c.u * rd[1] + c.v * rd[2] #offset = c.u * rd.x + c.v * rd.y
    Ray(c.origin + offset, normalize(c.lower_left_corner + s*c.horizontal +
                                     t*c.vertical - c.origin - offset))
end

"""
    Args:
        refraction_ratio: incident refraction index divided by refraction index of 
            hit surface. i.e. η/η′ in the figure above
"""
function refract(dir::SVector{3,Float32}, n⃗::SVector{3,Float32}, 
                 refraction_ratio::Float32)
    cosθ = min(-dir ⋅ n⃗, 1)
    r_out_perp = refraction_ratio * (dir + cosθ*n⃗)
    r_out_parallel = -√(abs(1-squared_length(r_out_perp))) * n⃗
    normalize(r_out_perp + r_out_parallel)
end


mutable struct Dielectric <: Material
    ir::Float32 # index of refraction, i.e. η.
end

function reflectance(cosθ::Float32, refraction_ratio::Float32)
    # Use Schlick's approximation for reflectance.
    # claforte: may be buggy? I'm getting black pixels in the Hollow Glass Sphere...
    r0 = (1f0-refraction_ratio) / (1f0+refraction_ratio)
    r0 = r0^2
    r0 + (1f0-r0)*((1f0-cosθ)^5)
end

function scatter(mat::Dielectric, r_in::Ray, rec::HitRecord)
    attenuation = @SVector[1f0,1f0,1f0]
    refraction_ratio = rec.front_face ? (1.0f0/mat.ir) : mat.ir # i.e. ηᵢ/ηₜ
    cosθ = min(-r_in.dir⋅rec.n⃗, 1.0f0)
    sinθ = √(1.0f0 - cosθ^2)
    cannot_refract = refraction_ratio * sinθ > 1.0
    if cannot_refract || reflectance(cosθ, refraction_ratio) > rand(Float32)
        dir = reflect(r_in.dir, rec.n⃗)
    else
        dir = refract(r_in.dir, rec.n⃗, refraction_ratio)
    end
    Scatter(Ray(rec.p, dir), attenuation) # TODO: rename reflected -> !absorbed?
end

"For debugging, represent vectors as RGB"
color_vec3_in_rgb(v::SVector{3,Float32}) = 0.5normalize(v) + @SVector[0.5f,0.5f,0.5f]

"""Compute color for a ray, recursively

    Args:
        depth: how many more levels of recursive ray bounces can we still compute?
"""
function ray_color(r::Ray, world::HittableList, depth=4)
    if depth <= 0
        return @SVector[0f0,0f0,0f0]
    end
        
    rec = hit(world, r, 1f-4, Inf32)
    if rec !== _no_hit
        # For debugging, represent vectors as RGB:
        # return color_vec3_in_rgb(rec.p) # show the normalized hit point
        # return color_vec3_in_rgb(rec.n⃗) # show the normal in RGB
        # return color_vec3_in_rgb(rec.p + rec.n⃗)
        # return color_vec3_in_rgb(random_vec3_in_sphere())
        # return color_vec3_in_rgb(rec.n⃗ + random_vec3_in_sphere())

        s = scatter(rec.mat, r, rec)
        if s.reflected
            return s.attenuation .* ray_color(s.r, world, depth-1)
        else
            return @SVector[0f0,0f0,0f0]
        end
    else
        skycolor(r)
    end
end

"""Render an image of `scene` using the specified camera, number of samples.

    Args:
        scene: a HittableList, e.g. a list of spheres
        n_samples: number of samples per pixel, eq. to C++ samples_per_pixel

    Equivalent to C++'s `main` function.
"""
function render(scene::HittableList, cam::Camera, image_width=400,
                n_samples=1)
    # Image
    aspect_ratio = 16.0f0/9.0f0 # TODO: use cam.aspect_ratio for consistency
    image_height = convert(Int64, floor(image_width / aspect_ratio))

    # Render
    img = zeros(RGB{Float32}, image_height, image_width)
    # Compared to C++, Julia is:
    # 1. column-major, i.e. iterate 1 column at a time, so invert i,j compared to C++
    # 2. 1-based, so no need to subtract 1 from image_width, etc.
    # 3. The array is Y-down, but `v` is Y-up 
    for i in 1:image_height, j in 1:image_width
        accum_color = @SVector[0f0,0f0,0f0]
        for s in 1:n_samples
            u = convert(Float32, j/image_width)
            v = convert(Float32, (image_height-i)/image_height) # i=Y-down, v=Y-up!
            if s != 1 # 1st sample is always centered, for 1-sample/pixel
                # claforte: I think the C++ version had a bug, the rand offset was
                # between [0,1] instead of centered at 0, e.g. [-0.5, 0.5].
                u += (rand(Float32) - 0.5f0) / image_width
                v += (rand(Float32) - 0.5f0) / image_height
            end
            ray = get_ray(cam, u, v)
            accum_color += ray_color(ray, scene)
        end
        img[i,j] = rgb_gamma2(accum_color / n_samples)
    end
    img
end


# Random spheres
function scene_random_spheres()::HittableList # dielectric spheres
    spheres = Sphere[]

    # ground 
    push!(spheres, Sphere(@SVector[0f0,-1000f0,-1f0], 1000f0, 
                          Lambertian(@SVector[0.5f0,0.5f0,0.5f0])))

    for a in -11:10, b in -11:10
        choose_mat = rand(Float32)
        center = @SVector[a + 0.9f0*rand(Float32), 0.2f0, b + 0.90f0*rand(Float32)]

        # skip spheres too close?
        if norm(center - @SVector[4f0,0.2f0,0f0]) < 0.9f0 continue end 
            
        if choose_mat < 0.8f0
            # diffuse
            albedo = @SVector[rand(Float32) for i ∈ 1:3] .* @SVector[rand(Float32) for i ∈ 1:3]
            push!(spheres, Sphere(center, 0.2f0, Lambertian(albedo)))
        elseif choose_mat < 0.95f0
            # metal
            albedo = @SVector[random_between(0.5f0,1.0f0) for i ∈ 1:3]
            fuzz = random_between(0.0f0, 5.0f0)
            push!(spheres, Sphere(center, 0.2f0, Metal(albedo, fuzz)))
        else
            # glass
            push!(spheres, Sphere(center, 0.2f0, Dielectric(1.5f0)))
        end
    end

    push!(spheres, Sphere(@SVector[0f0,1f0,0f0], 1.0f0, Dielectric(1.5f0)))
    push!(spheres, Sphere(@SVector[-4f0,1f0,0f0], 1.0f0, 
                          Lambertian(@SVector[0.4f0,0.2f0,0.1f0])))
    push!(spheres, Sphere(@SVector[4f0,1f0,0f0], 1.0f0, 
                          Metal(@SVector[0.7f0,0.6f0,0.5f0], 0.0f0)))
    HittableList(spheres)
end


t_lookfrom1 = @SVector[13.0f0,2.0f0,3.0f0]
t_lookat1 = @SVector[0.0f0,0.0f0,0.0f0]
t_cam1 = default_camera(
    t_lookfrom1, 
    t_lookat1, 
    @SVector[0.0f0,1.0f0,0.0f0], 
    20.0f0, 16.0f0/9.0f0, 0.1f0, 10.0f0
)

scene = scene_random_spheres()
render(scene, t_cam1, 200, 43)
#=
@btime render(scene, t_cam1, 200, 43);
  10.222 s (2399087 allocations: 110.08 MiB)
=#