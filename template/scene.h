#pragma once

// -----------------------------------------------------------
// scene.h
// Simple test scene for ray tracing experiments. Goals:
// - Super-fast scene intersection
// - Easy interface: scene.FindNearest / IsOccluded
// - With normals and albedo: GetNormal / GetAlbedo
// - Area light source (animated), for light transport
// - Primitives can be hit from inside - for dielectrics
// - Can be extended with other primitives and/or a BVH
// - Optionally animated - for temporal experiments
// - Not everything is axis aligned - for cache experiments
// - Can be evaluated at arbitrary time - for motion blur
// - Has some high-frequency details - for filtering
// Some speed tricks that severely affect maintainability
// are enclosed in #ifdef SPEEDTRIX / #endif. Mind these
// if you plan to alter the scene in any way.
// -----------------------------------------------------------

#define SPEEDTRIX

#define PLANE_X(o,i) {if((t=-(ray.O.x+o)*ray.rD.x)<ray.t)ray.t=t,ray.objIdx=i;}
#define PLANE_Y(o,i) {if((t=-(ray.O.y+o)*ray.rD.y)<ray.t)ray.t=t,ray.objIdx=i;}
#define PLANE_Z(o,i) {if((t=-(ray.O.z+o)*ray.rD.z)<ray.t)ray.t=t,ray.objIdx=i;}

namespace Tmpl8 {

__declspec(align(64)) class Ray
{
public:
	Ray() = default;
	Ray( float3 origin, float3 direction, float distance = 1e34f )
	{
		O = origin, D = direction, t = distance;
		// calculate reciprocal ray direction for triangles and AABBs
		rD = float3( 1 / D.x, 1 / D.y, 1 / D.z );
	#ifdef SPEEDTRIX
		d0 = d1 = d2 = 0;
	#endif
	}
	float3 IntersectionPoint() { return O + t * D; }
	// ray data
#ifndef SPEEDTRIX
	float3 O, D, rD;
#else
	union { struct { float3 O; float d0; }; __m128 O4; };
	union { struct { float3 D; float d1; }; __m128 D4; };
	union { struct { float3 rD; float d2; }; __m128 rD4; };
#endif
	float t = 1e34f;
	int objIdx = -1;
	bool inside = false; // true when in medium
};

class Material
{
public:
	Material() = default;
	Material(
	bool isEmissive,        // 是否发光
	float3 color = float3(0, 0, 0),    // 颜色
	double specularRate = 0.0f,     // 反射光占比
	double roughness = 1.0f,        // 粗糙程度
	double refractRate = 0.0f,      // 折射光占比
	double refractAngle = 1.0f,     // 折射率
	double refractRoughness = 0.0f // 折射粗糙度
	) : isEmissive(isEmissive),
		color(color),
		specularRate(specularRate),
		roughness(roughness),
		refractRate(refractRate),
		refractAngle(refractAngle),
		refractRoughness(refractRoughness)
	{

	}

	bool isEmissive;        // 是否发光
	float3 color;     // 颜色
	double specularRate;     // 反射光占比
	double roughness;        // 粗糙程度
	double refractRate;      // 折射光占比
	double refractAngle;     // 折射率
	double refractRoughness; // 折射粗糙度
};

inline float3 mix(float3 x, float3 y, float a) {
	return (1 - a) * x + a * y;
}

inline float3 random_in_uint_sphere() {
	while (true) {
		float3 result;
		float x = rand() * (2.0 / RAND_MAX) - 1.0;
		float y = rand() * (2.0 / RAND_MAX) - 1.0;
		float z = rand() * (2.0 / RAND_MAX) - 1.0;
		result = float3(x, y, z);
		if (length(result) <= 1.0) return result;
	}
}

class Shape {
public:
	Shape() = default;
	Shape(Material mat) : material(mat) {}

	float3 GetColor() const {
		return material.color;
	}

	Material material;
};

class Sphere : public Shape
{
public:
	Sphere() = default;
	Sphere(int idx, float3 p, float r, Material mat) :
		Shape(mat), pos(p), r2(r* r), invr(1 / r), objIdx(idx) {}
	void Intersect(Ray& ray) const
	{
		float3 oc = ray.O - this->pos;
		float b = dot(oc, ray.D);
		float c = dot(oc, oc) - this->r2;
		float t, d = b * b - c;
		if (d <= 0) return;
		d = sqrtf(d), t = -b - d;
		if (t < ray.t && t > 0)
		{
			ray.t = t, ray.objIdx = objIdx;
			return;
		}
		t = d - b;
		if (t < ray.t && t > 0)
		{
			ray.t = t, ray.objIdx = objIdx;
			return;
		}
	}
	float3 GetNormal(const float3 I) const
	{
		return (I - this->pos) * invr;
	}
	float3 GetAlbedo(const float3 I) const
	{
		return float3(0.93f);
	}
	float3 pos = 0;
	float r2 = 0, invr = 0;
	int objIdx = -1;
};

// -----------------------------------------------------------
// Plane primitive
// Basic infinite plane, defined by a normal and a Diffuse
// from the origin (in the direction of the normal).
// -----------------------------------------------------------
class Plane : public Shape
{
public:
	Plane() = default;
	Plane(int idx, float3 normal, float dist, Material mat) :
		Shape(mat), N(normal), d(dist), objIdx(idx) {}
	void Intersect(Ray& ray) const
	{
		float t = -(dot(ray.O, this->N) + this->d) / (dot(ray.D, this->N));
		if (t < ray.t && t > 0) ray.t = t, ray.objIdx = objIdx;
	}
	float3 GetNormal(const float3 I) const
	{
		return N;
	}
	float3 GetAlbedo(const float3 I) const
	{
		if (N.y == 1)
		{
			// floor albedo: checkerboard
			int ix = (int)(I.x * 2 + 96.01f);
			int iz = (int)(I.z * 2 + 96.01f);
			// add deliberate aliasing to two tile
			if (ix == 98 && iz == 98) ix = (int)(I.x * 32.01f), iz = (int)(I.z * 32.01f);
			if (ix == 94 && iz == 98) ix = (int)(I.x * 64.01f), iz = (int)(I.z * 64.01f);
			return float3(((ix + iz) & 1) ? 1 : 0.3f);
		}
		else if (N.z == -1)
		{
			// back wall: logo
			static Surface logo("assets/logo.png");
			int ix = (int)((I.x + 4) * (128.0f / 8));
			int iy = (int)((2 - I.y) * (64.0f / 3));
			uint p = logo.pixels[(ix & 127) + (iy & 63) * 128];
			uint3 i3((p >> 16) & 255, (p >> 8) & 255, p & 255);
			return float3(i3) * (1.0f / 255.0f);
		}
		return float3(0.93f);
	}
	float3 N;
	float d;
	int objIdx = -1;
};

// -----------------------------------------------------------
// Cube primitive
// Oriented cube. Unsure if this will also work for rays that
// start inside it; maybe not the best candidate for testing
// dielectrics.
// -----------------------------------------------------------
class Cube : public Shape
{
public:
	Cube() = default;
	Cube(int idx, float3 pos, float3 size, mat4 transform = mat4::Identity(), Material mat = Material()) :
		Shape(mat)
	{
		objIdx = idx;
		b[0] = pos - 0.5f * size, b[1] = pos + 0.5f * size;
		M = transform, invM = transform.FastInvertedTransformNoScale();
	}
	void Intersect(Ray& ray) const
	{
		// 'rotate' the cube by transforming the ray into object space
		// using the inverse of the cube transform.
		float3 O = TransformPosition(ray.O, invM);
		float3 D = TransformVector(ray.D, invM);
		float rDx = 1 / D.x, rDy = 1 / D.y, rDz = 1 / D.z;
		int signx = D.x < 0, signy = D.y < 0, signz = D.z < 0;
		float tmin = (b[signx].x - O.x) * rDx;
		float tmax = (b[1 - signx].x - O.x) * rDx;
		float tymin = (b[signy].y - O.y) * rDy;
		float tymax = (b[1 - signy].y - O.y) * rDy;
		if (tmin > tymax || tymin > tmax) return;
		tmin = max(tmin, tymin), tmax = min(tmax, tymax);
		float tzmin = (b[signz].z - O.z) * rDz;
		float tzmax = (b[1 - signz].z - O.z) * rDz;
		if (tmin > tzmax || tzmin > tmax) return;
		tmin = max(tmin, tzmin), tmax = min(tmax, tzmax);
		if (tmin > 0)
		{
			if (tmin < ray.t) ray.t = tmin, ray.objIdx = objIdx;
		}
		else if (tmax > 0)
		{
			if (tmax < ray.t) ray.t = tmax, ray.objIdx = objIdx;
		}
	}
	float3 GetNormal(const float3 I) const
	{
		// transform intersection point to object space
		float3 objI = TransformPosition(I, invM);
		// determine normal in object space
		float3 N = float3(-1, 0, 0);
		float d0 = fabs(objI.x - b[0].x), d1 = fabs(objI.x - b[1].x);
		float d2 = fabs(objI.y - b[0].y), d3 = fabs(objI.y - b[1].y);
		float d4 = fabs(objI.z - b[0].z), d5 = fabs(objI.z - b[1].z);
		float minDist = d0;
		if (d1 < minDist) minDist = d1, N.x = 1;
		if (d2 < minDist) minDist = d2, N = float3(0, -1, 0);
		if (d3 < minDist) minDist = d3, N = float3(0, 1, 0);
		if (d4 < minDist) minDist = d4, N = float3(0, 0, -1);
		if (d5 < minDist) minDist = d5, N = float3(0, 0, 1);
		// return normal in world space
		return TransformVector(N, M);
	}
	float3 GetAlbedo(const float3 I) const
	{
		return float3(1, 1, 1);
	}
	float3 b[2];
	mat4 M, invM;
	int objIdx = -1;
};

// -----------------------------------------------------------
// Quad primitive
// Oriented quad, intended to be used as a light source.
// -----------------------------------------------------------
class Quad : public Shape
{
public:
	Quad() = default;
	Quad(int idx, float s, mat4 transform = mat4::Identity(), Material mat = Material()) :
		Shape(mat)
	{
		objIdx = idx;
		size = s * 0.5f;
		T = transform, invT = transform.FastInvertedTransformNoScale();
	}
	void Intersect(Ray& ray) const
	{
		const float3 O = TransformPosition(ray.O, invT);
		const float3 D = TransformVector(ray.D, invT);
		const float t = O.y / -D.y;
		if (t < ray.t && t > 0)
		{
			float3 I = O + t * D;
			if (I.x > -size && I.x < size && I.z > -size && I.z < size)
				ray.t = t, ray.objIdx = objIdx;
		}
	}
	float3 GetNormal(const float3 I) const
	{
		// TransformVector( float3( 0, -1, 0 ), T ) 
		return float3(-T.cell[1], -T.cell[5], -T.cell[9]);
	}
	float3 GetAlbedo(const float3 I) const
	{
		return float3(10);
	}
	float size;
	mat4 T, invT;
	int objIdx = -1;
};

class Triangle : public Shape
{
public:

	Triangle() = default;
	Triangle(int idx, float3 v0, float3 v1, float3 v2, float3 color, Material mat) :
		Shape(mat), objIdx(idx), v0(v0), v1(v1), v2(v2)
	{
		N = normalize(cross(v1 - v0, v2 - v0));
	};

	void Intersect(Ray& ray) const
	{
		float3 O = TransformPosition(ray.O, invM);
		float3 D = TransformVector(ray.D, invM);
		float DP = dot(N, v0);

		// parralel check
		float parallel = dot(N, D);
		if (parallel == 0) return;
		float t = -(dot(N, O) + DP) / dot(N, D);
		if (t < ray.t && t > 0) {
			float3 I = O + D * ray.t;

			// edge 0
			float3 edge0 = v1 - v0;
			float3 vp0 = I - v0;
			float3 C = cross(edge0, vp0);
			if (dot(N, C) < 0) return;

			// edge 1
			float3 edge1 = v2 - v1;
			float3 vp1 = I - v1;
			C = cross(edge1, vp1);
			if (dot(N, C) < 0)  return;

			// edge 2
			float3 edge2 = v0 - v2;
			float3 vp2 = I - v2;
			C = cross(edge2, vp2);
			if (dot(N, C) < 0) return;

			ray.t = t;
			ray.objIdx = objIdx;

		}
	}

	float3 GetNormal() const
	{
		return N;
	}

	float3 v0, v1, v2;
	float3 N;
	int objIdx = -1;
	mat4 M, invM;
};

// -----------------------------------------------------------
// Scene class
// We intersect this. The query is internally forwarded to the
// list of primitives, so that the nearest hit can be returned.
// For this hit (distance, obj id), we can query the normal and
// albedo.
// -----------------------------------------------------------
class Scene
{
public:
	Scene()
	{
		Material light_source = Material(true, float3(1.0));
		Material white = Material(false, float3(1.0));
		Material red = Material(false, float3(1.0, 0, 0));
		Material green = Material(false, float3(0, 1.0, 0));
		Material blue = Material(false, float3(0, 0, 1.0));
		Material yellow = Material(false, float3(1.0, 1.0, 0), 0.3, 0.1);
		Material purple = Material(false, float3(1.0, 0, 1.0), 0.3, 1.0, 0.95, 0.1, 0);
		// we store all primitives in one continuous buffer
		quad = Quad(0, 1, mat4::Identity(), light_source);									// 0: light source
		sphere = Sphere( 1, float3( 0 ), 0.5f, blue);				// 1: bouncing ball
		sphere2 = Sphere( 2, float3( 0, 2.5f, -3.07f ), 8, yellow);	// 2: rounded corners
		cube = Cube( 3, float3( 0 ), float3( 1.15f ), mat4::Identity(), purple);			// 3: cube
		plane[0] = Plane( 4, float3( 1, 0, 0 ), 3, red );			// 4: left wall
		plane[1] = Plane( 5, float3( -1, 0, 0 ), 2.99f, blue );		// 5: right wall
		plane[2] = Plane( 6, float3( 0, 1, 0 ), 1, white );			// 6: floor
		plane[3] = Plane( 7, float3( 0, -1, 0 ), 2, white );			// 7: ceiling
		plane[4] = Plane( 8, float3( 0, 0, 1 ), 3, white );			// 8: front wall
		plane[5] = Plane( 9, float3( 0, 0, -1 ), 3.99f, white );		// 9: back wall
		shapes.push_back(&quad);
		shapes.push_back(&sphere);
		shapes.push_back(&sphere2);
		shapes.push_back(&cube);
		shapes.push_back(&plane[0]);
		shapes.push_back(&plane[1]);
		shapes.push_back(&plane[2]);
		shapes.push_back(&plane[3]);
		shapes.push_back(&plane[4]);
		shapes.push_back(&plane[5]);
		SetTime( 0 );
		// Note: once we have triangle support we should get rid of the class
		// hierarchy: virtuals reduce performance somewhat.
	}
	void SetTime( float t )
	{
		// default time for the scene is simply 0. Updating/ the time per frame 
		// enables animation. Updating it per ray can be used for motion blur.
		//animTime = t;
		//// light source animation: swing
		//mat4 M1base = mat4::Translate( float3( 0, 2.6f, 2 ) );
		//mat4 M1 = M1base * mat4::RotateZ( sinf( animTime * 0.6f ) * 0.1f ) * mat4::Translate( float3( 0, -0.9, 0 ) );
		//quad.T = M1, quad.invT = M1.FastInvertedTransformNoScale();
		//// cube animation: spin
		//mat4 M2base = mat4::RotateX( PI / 4 ) * mat4::RotateZ( PI / 4 );
		//mat4 M2 = mat4::Translate( float3( 1.4f, 0, 2 ) ) * mat4::RotateY( animTime * 0.5f ) * M2base;
		//cube.M = M2, cube.invM = M2.FastInvertedTransformNoScale();
		//// sphere animation: bounce
		//float tm = 1 - sqrf( fmodf( animTime, 2.0f ) - 1 );
		//sphere.pos = float3( -1.4f, -0.5f + tm, 2 );

		mat4 M1base = mat4::Translate(float3(0, 2.6f, 2));
		mat4 M1 = M1base * mat4::Translate(float3(0, -0.9, 0));

		quad.T = M1, quad.invT = M1.FastInvertedTransformNoScale();

		mat4 M2 = mat4::Translate(float3(1.4f, 0, 2));
		cube.M = M2, cube.invM = M2.FastInvertedTransformNoScale();

		sphere.pos = float3(-1.4f, 0.2, 2);
	}
	float3 GetLightPos() const
	{
		// light point position is the middle of the swinging quad
		float3 corner1 = TransformPosition( float3( -0.5f, 0, -0.5f ), quad.T );
		float3 corner2 = TransformPosition( float3( 0.5f, 0, 0.5f ), quad.T );
		return (corner1 + corner2) * 0.5f - float3( 0, 0.01f, 0 );
	}
	float3 GetLightColor(int objIdx) const
	{
		if (objIdx == -1) return float3(0); // or perhaps we should just crash
		return shapes[objIdx]->material.color;
	}
	Material GetMaterial(int objIdx) const {
		return shapes[objIdx]->material;
	}
	void FindNearest( Ray& ray ) const
	{
		// room walls - ugly shortcut for more speed
		float t;
		if (ray.D.x < 0) PLANE_X( 3, 4 ) else PLANE_X( -2.99f, 5 );
		if (ray.D.y < 0) PLANE_Y( 1, 6 ) else PLANE_Y( -2, 7 );
		if (ray.D.z < 0) PLANE_Z( 3, 8 ) else PLANE_Z( -3.99f, 9 );
		quad.Intersect( ray );
		sphere.Intersect( ray );
		sphere2.Intersect( ray );
		cube.Intersect( ray );
	}
	bool IsOccluded( Ray& ray ) const
	{
		float rayLength = ray.t;
		// skip planes: it is not possible for the walls to occlude anything
		quad.Intersect( ray );
		sphere.Intersect( ray );
		sphere2.Intersect( ray );
		cube.Intersect( ray );
		return ray.t < rayLength;
		// technically this is wasteful: 
		// - we potentially search beyond rayLength
		// - we store objIdx and t when we just need a yes/no
		// - we don't 'early out' after the first occlusion
	}
	float3 GetNormal( int objIdx, float3 I, float3 wo ) const
	{
		// we get the normal after finding the nearest intersection:
		// this way we prevent calculating it multiple times.
		if (objIdx == -1) return float3( 0 ); // or perhaps we should just crash
		float3 N;
		if (objIdx == 0) N = quad.GetNormal( I );
		else if (objIdx == 1) N = sphere.GetNormal( I );
		else if (objIdx == 2) N = sphere2.GetNormal( I );
		else if (objIdx == 3) N = cube.GetNormal( I );
		else 
		{
			// faster to handle the 6 planes without a call to GetNormal
			N = float3( 0 );
			N[(objIdx - 4) / 2] = 1 - 2 * (float)(objIdx & 1);
		}
		if (dot( N, wo ) > 0) N = -N; // hit backside / inside
		return N;
	}
	float3 GetAlbedo( int objIdx, float3 I ) const
	{
		if (objIdx == -1) return float3( 0 ); // or perhaps we should just crash
		if (objIdx == 0) return quad.GetAlbedo( I );
		if (objIdx == 1) return sphere.GetAlbedo( I );
		if (objIdx == 2) return sphere2.GetAlbedo( I );
		if (objIdx == 3) return cube.GetAlbedo( I );
		return plane[objIdx - 4].GetAlbedo( I );
		// once we have triangle support, we should pass objIdx and the bary-
		// centric coordinates of the hit, instead of the intersection location.
	}
	float GetReflectivity( int objIdx, float3 I ) const
	{
		if (objIdx == 1 /* ball */) return 1;
		if (objIdx == 6 /* floor */) return 0.3f;
		return 0;
	}
	float GetRefractivity( int objIdx, float3 I ) const
	{
		return objIdx == 3 ? 1.0f : 0.0f;
	}
	__declspec(align(64)) // start a new cacheline here
	float animTime = 0;
	Quad quad;
	Sphere sphere;
	Sphere sphere2;
	Cube cube;
	Plane plane[6];
	vector<Shape*> shapes;
};

}