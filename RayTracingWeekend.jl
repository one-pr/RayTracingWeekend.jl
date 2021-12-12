using Images
using StaticArrays
using LinearAlgebra

const Vec3{T<:AbstractFloat} = SVector{3, T}


squared_length(v::Vec3) = v ⋅ v
near_zero(v::Vec3) = squared_length(v) < 1e-5


#= Manipulating images =#
rgb(v::Vec3) = RGB(v[1], v[2], v[3]) # IIRC RGB(v...) was much slower...
rgb_gamma2(v::Vec3) = RGB(sqrt.(v)...) # claforte: try to expand?

struct Ray{T}
    origin::Vec3{T}
    dir::Vec3{T} # direction (unit vector)
end

# equivalent to C++'s ray.at()
"3D Point of ray `r` evaluated at parameter t"
point(r::Ray{T}, t::T) where T <: AbstractFloat = r.origin .+ t .* r.dir


#= Random vectors =#

# equiv to random_double()
random_between(min::T=0.0, max::T=1.0) where T = rand(T)*(max-min) + min
random_vec3(min::T, max::T) where T = @SVector [random_between(min,max) for i∈1:3]
random_vec2(min::T, max::T) where T = @SVector [random_between(min,max) for i∈1:2]
function random_vec3_in_sphere(::Type{T}) where T # equiv to random_in_unit_sphere()
    while true
        p = random_vec3(T(-1), T(1))
        if p⋅p <= 1
            return p
        end
    end
end

"Random unit vector. Equivalent to C++'s `unit_vector(random_in_unit_sphere())`"
random_vec3_on_sphere(::Type{T}) where T = normalize(random_vec3_in_sphere(T))
function random_vec2_in_disk(::Type{T}) where T # equiv to random_in_unit_disk()
    while true
        p = random_vec2(T(-1), T(1))
        if p⋅p <= 1
            return p
        end
    end
end


#= Rays, simple camera, and background =#

function skycolor(ray::Ray{T}) where T
    # NOTE: unlike in the C++ implem., we assume the ray direction is pre-normalized.
    white = @SVector T[1.0, 1.0, 1.0]
    skyblue = @SVector T[0.5, 0.7, 1.0]
    t = T(0.5)*(ray.dir[2] + one(T))
    (one(T)-t)*white + t*skyblue
end


"An object that can be hit by Ray"
abstract type Hittable end

"""Materials tell us how rays interact with a surface"""
abstract type Material{T<: AbstractFloat} end


"Record a hit between a ray and an object's surface"
mutable struct HitRecord{T<: AbstractFloat}
    t::T # vector from the ray's origin to the intersection with a surface. 

    # If t==Inf32, there was no hit, and all following values are undefined!
    #
    p::Vec3{T} # point of intersection between an object's surface and ray
    n⃗::Vec3{T} # local normal (see diagram below)

    # If true, our ray hit from outside to the front of the surface. 
    # If false, the ray hit from within.
    front_face::Bool
    mat::Material{T}

    HitRecord{T}() where T = new{T}(typemax(T)) # no hit!
    HitRecord(t::T,p,n⃗,front_face,mat) where T = new{T}(t,p,n⃗,front_face,mat)
end

struct Sphere{T<: AbstractFloat} <: Hittable
    center::Vec3{T}
    radius::T
    mat::Material{T}
end


"""Equivalent to `hit_record.set_face_normal()`"""
function ray_to_HitRecord(t::T, p, outward_n⃗, r_dir::Vec3{T}, mat::Material{T}) where T
    front_face = r_dir ⋅ outward_n⃗ < 0
    n⃗ = front_face ? outward_n⃗ : -outward_n⃗
    HitRecord(t,p,n⃗,front_face,mat)
end

struct Scatter{T<: AbstractFloat}
    r::Ray{T}
    attenuation::Vec3{T}

    # claforte: TODO: rename to "absorbed?", i.e. not reflected/refracted?
    reflected::Bool # whether the scattered ray was reflected, or fully absorbed
    Scatter(r::Ray{T}, a::Vec3{T}, reflected=true) where T = new{T}(r,a,reflected)
end

"Diffuse material"
mutable struct Lambertian{T} <: Material{T}
    albedo::Vec3{T}
end

"""Compute reflection vector for v (pointing to surface) and normal n⃗.

    See [diagram](https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection)
"""
reflect(v::Vec3{T}, n⃗::Vec3{T}) where T = v - (2v⋅n⃗)*n⃗

"""Create a scattered ray emitted by `mat` from incident Ray `r`. 

    Args:
        rec: the HitRecord of the surface from which to scatter the ray.

    Return missing if it's fully absorbed.
"""
function scatter(mat::Lambertian{T}, r::Ray{T}, rec::HitRecord{T})::Scatter{T} where T
    scatter_dir = rec.n⃗ + random_vec3_on_sphere(T)
    if near_zero(scatter_dir) # Catch degenerate scatter direction
        scatter_dir = rec.n⃗ 
    else
        scatter_dir = normalize(scatter_dir)
    end
    scattered_r = Ray{T}(rec.p, scatter_dir)
    attenuation = mat.albedo
    return Scatter(scattered_r, attenuation)
end

const _no_hit = HitRecord{Float64}() # will be reused 

function hit(s::Sphere{T}, r::Ray{T}, tmin::T, tmax::T) where T
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

const HittableList = Vector{Hittable}

"""Find closest hit between `Ray r` and a list of Hittable objects `h`, within distance `tmin` < `tmax`"""
function hit(hittables::HittableList, r::Ray{T}, tmin::T, tmax::T) where T
    closest = tmax # closest t so far
    rec = _no_hit
    for h in hittables
        temprec = hit(h, r, tmin, closest)
        if temprec !== _no_hit
            rec = temprec
            closest = rec.t # i.e. ignore any further hit > this one's.
        end
    end
    rec
end


# Metal material

mutable struct Metal{T} <: Material{T}
    albedo::Vec3{T}
    fuzz::T # how big the sphere used to generate fuzzy reflection rays. 0=none
    Metal(a::Vec3{T}, f::T=0.0) where T = new{T}(a,f)
end

function scatter(mat::Metal{T}, r_in::Ray{T}, rec::HitRecord)::Scatter{T} where T
    reflected = normalize(reflect(r_in.dir, rec.n⃗) + mat.fuzz*random_vec3_on_sphere(T))
    Scatter(Ray(rec.p, reflected), mat.albedo)
end

mutable struct Camera{T<: AbstractFloat}
    origin::Vec3{T}
    lower_left_corner::Vec3{T}
    horizontal::Vec3{T}
    vertical::Vec3{T}
    u::Vec3{T}
    v::Vec3{T}
    w::Vec3{T}
    lens_radius::T
end

"""
    Args:
        vfov: vertical field-of-view in degrees
        aspect_ratio: horizontal/vertical ratio of pixels
      aperture: if 0 - no depth-of-field
"""
function default_camera(lookfrom::Vec3{T}=[0,0,0], 
            lookat::Vec3{T}=[0,0,-1], 
            vup::Vec3{T}=[0,1,0], vfov::T=90.0, aspect_ratio::T=16/9,
                        aperture::T=0.0, focus_dist::T=1.0) where T
    viewport_height = T(2) * tand(vfov/T(2))
    viewport_width = aspect_ratio * viewport_height

    w = normalize(lookfrom - lookat)
    u = normalize(vup × w)
    v = w × u

    origin = lookfrom
    horizontal = focus_dist * viewport_width * u
    vertical = focus_dist * viewport_height * v
    lower_left_corner = origin - horizontal/T(2) - vertical/T(2) - focus_dist*w
    lens_radius = aperture/T(2)
    Camera{T}(origin, lower_left_corner, horizontal, vertical, u, v, w, lens_radius)
end

default_camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist; elem_type::Type{T}) where T = 
    default_camera(
        Vec3{T}(lookfrom), Vec3{T}(lookat), Vec3{T}(vup),
        T(vfov), T(aspect_ratio), T(aperture), T(focus_dist)
    )

function get_ray(c::Camera{T}, s::T, t::T) where T
    rd = SVector{2, T}(c.lens_radius * random_vec2_in_disk(T))
    offset = c.u * rd[1] + c.v * rd[2] #offset = c.u * rd.x + c.v * rd.y
    Ray(c.origin + offset, normalize(c.lower_left_corner + s*c.horizontal +
                                     t*c.vertical - c.origin - offset))
end

"""
    Args:
        refraction_ratio: incident refraction index divided by refraction index of 
            hit surface. i.e. η/η′ in the figure above
"""
function refract(dir::Vec3{T}, n⃗::Vec3{T}, 
                 refraction_ratio::T) where T
    cosθ = min(-dir ⋅ n⃗, one(T))
    r_out_perp = refraction_ratio * (dir + cosθ*n⃗)
    r_out_parallel = -√(abs(one(T)-squared_length(r_out_perp))) * n⃗
    normalize(r_out_perp + r_out_parallel)
end


mutable struct Dielectric{T} <: Material{T}
    ir::T # index of refraction, i.e. η.
end

function reflectance(cosθ, refraction_ratio)
    # Use Schlick's approximation for reflectance.
    # claforte: may be buggy? I'm getting black pixels in the Hollow Glass Sphere...
    r0 = (1-refraction_ratio) / (1+refraction_ratio)
    r0 = r0^2
    r0 + (1-r0)*((1-cosθ)^5)
end

function scatter(mat::Dielectric{T}, r_in::Ray{T}, rec::HitRecord{T}) where T
    attenuation = @SVector T[1,1,1]
    refraction_ratio = rec.front_face ? (one(T)/mat.ir) : mat.ir # i.e. ηᵢ/ηₜ
    cosθ = min(-r_in.dir⋅rec.n⃗, one(T))
    sinθ = √(one(T) - cosθ^2)
    cannot_refract = refraction_ratio * sinθ > one(T)
    if cannot_refract || reflectance(cosθ, refraction_ratio) > rand(T)
        dir = reflect(r_in.dir, rec.n⃗)
    else
        dir = refract(r_in.dir, rec.n⃗, refraction_ratio)
    end
    Scatter(Ray{T}(rec.p, dir), attenuation) # TODO: rename reflected -> !absorbed?
end

"For debugging, represent vectors as RGB"
color_vec3_in_rgb(v::Vec3{T}) where T = T(0.5)*normalize(v) + @SVector T[0.5,0.5,0.5]

"""Compute color for a ray, recursively

    Args:
        depth: how many more levels of recursive ray bounces can we still compute?
"""
function ray_color(r::Ray{T}, world::HittableList, depth=4) where T
    if depth <= 0
        return @SVector T[0, 0, 0]
    end

    rec = hit(world, r, T(1e-4), typemax(T))
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
            return @SVector T[0, 0, 0]
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
function render(scene::HittableList, cam::Camera{T}, image_width=400,
                n_samples=1) where T
    # Image
    aspect_ratio = T(16.0/9.0) # TODO: use cam.aspect_ratio for consistency
    image_height = convert(Int64, floor(image_width / aspect_ratio))

    # Render
    img = zeros(RGB{T}, image_height, image_width)
    # Compared to C++, Julia is:
    # 1. column-major, i.e. iterate 1 column at a time, so invert i,j compared to C++
    # 2. 1-based, so no need to subtract 1 from image_width, etc.
    # 3. The array is Y-down, but `v` is Y-up 
    for i in 1:image_height, j in 1:image_width
        accum_color = @SVector T[0,0,0]
        for s in 1:n_samples
            u = convert(T, j/image_width)
            v = convert(T, (image_height-i)/image_height) # i=Y-down, v=Y-up!
            if s != 1 # 1st sample is always centered, for 1-sample/pixel
                # claforte: I think the C++ version had a bug, the rand offset was
                # between [0,1] instead of centered at 0, e.g. [-0.5, 0.5].
                u += (rand(T) - T(0.5)) / image_width
                v += (rand(T) - T(0.5)) / image_height
            end
            ray = get_ray(cam, u, v)
            accum_color += ray_color(ray, scene)
        end
        img[i,j] = rgb_gamma2(accum_color / n_samples)
    end
    img
end


# Random spheres
function scene_random_spheres(; elem_type::Type{T}) where T # dielectric spheres
    spheres = Sphere[]

    # ground 
    push!(spheres, Sphere((@SVector T[0,-1000,-1]), T(1000), 
                          Lambertian(@SVector T[0.5,0.5,0.5])))

    for a in -11:10, b in -11:10
        choose_mat = rand(T)
        center = @SVector [a + T(0.9)*rand(T), T(0.2), b + T(0.9)*rand(T)]

        # skip spheres too close?
        if norm(center - @SVector T[4,0.2,0]) < 0.9 continue end 

        if choose_mat < 0.8
            # diffuse
            albedo = @SVector[rand(T) for i ∈ 1:3] .* @SVector[rand(T) for i ∈ 1:3]
            push!(spheres, Sphere(center, T(0.2), Lambertian(albedo)))
        elseif choose_mat < 0.95
            # metal
            albedo = @SVector[random_between(T(0.5), T(1.0)) for i ∈ 1:3]
            fuzz = random_between(T(0.0), T(5.0))
            push!(spheres, Sphere(center, T(0.2), Metal(albedo, fuzz)))
        else
            # glass
            push!(spheres, Sphere(center, T(0.2), Dielectric(T(1.5))))
        end
    end

    push!(spheres, Sphere((@SVector T[0,1,0]), one(T), Dielectric(T(1.5))))
    push!(spheres, Sphere((@SVector T[-4,1,0]), one(T), 
                          Lambertian(@SVector T[0.4,0.2,0.1])))
    push!(spheres, Sphere((@SVector T[4,1,0]), one(T), 
                          Metal((@SVector T[0.7,0.6,0.5]), zero(T))))
    HittableList(spheres)
end

# Float32 / Float64
ELEM_TYPE = Float32
t_cam1 = default_camera(
    [13, 2, 3], 
    [0, 0, 0], 
    [0, 1, 0], 
    20, 16/9, 0.1, 10,
    ; elem_type=ELEM_TYPE
)

scene = scene_random_spheres(; elem_type=ELEM_TYPE)
render(scene, t_cam1, 200, 43)
#=
@btime render(scene, t_cam1, 200, 43);
  10.222 s (2399087 allocations: 110.08 MiB)

Float32
@btime render(scene, t_cam1, 200, 43);
  10.756 s (7370438 allocations: 188.79 MiB)
Float64
@btime render(scene, t_cam1, 200, 43);
  8.797 s (2274922 allocations: 174.08 MiB)
=#