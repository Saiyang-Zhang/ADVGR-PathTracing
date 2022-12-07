#include "precomp.h"

// -----------------------------------------------------------
// Initialize the renderer
// -----------------------------------------------------------
void Renderer::Init()
{
	// create fp32 rgb pixel buffer to render to
	accumulator = (float4*)MALLOC64( SCRWIDTH * SCRHEIGHT * 16 );
	memset( accumulator, 0, SCRWIDTH * SCRHEIGHT * 16 );
	sample = 1.0;
}

// -----------------------------------------------------------
// Evaluate light transport
// -----------------------------------------------------------
float3 Renderer::Trace( Ray& ray, int sample = 256 )
{
	scene.FindNearest( ray );
	if (ray.objIdx == -1) return 0; // or a fancy sky color
	float3 I = ray.O + ray.t * ray.D;
	float3 N = scene.GetNormal( ray.objIdx, I, ray.D );
	Material mat = scene.GetMaterial(ray.objIdx);
	//float3 albedo = scene.GetAlbedo( ray.objIdx, I );
	if (mat.isEmissive) return mat.color;

	float3 color = float3(0);

	float r;
	r = rand() * (1.0 / RAND_MAX);
	Ray randomRay = Ray(I, random_in_uint_sphere());

	if (r < mat.specularRate)
	{
		float3 reflectRayDir = normalize(reflect(ray.D, N));
		randomRay.D = mix(reflectRayDir, randomRay.D, mat.roughness);
		color += PathTrace(randomRay, 0);
	}
	else if (mat.specularRate <= r && r <= mat.refractRate)
	{
		float3 refractRayDir = normalize(refract(ray.D, N, float(mat.refractAngle)));
		randomRay.D = mix(refractRayDir, -randomRay.D, mat.refractRoughness);
		color += PathTrace(randomRay, 0);
	}
	else
	{
		float3 srcColor = mat.color;
		float3 ptColor = PathTrace(randomRay, 0);
		color += ptColor * srcColor;
	}

	return color;
}

float3 Renderer::PathTrace(Ray& ray, int iter = 0)
{
	if (iter > 5) return float3(0);
	scene.FindNearest(ray);
	if (ray.objIdx == -1) return 0; // or a fancy sky color
	Material mat = scene.GetMaterial(ray.objIdx);

	if (mat.isEmissive) return mat.color;
	float3 I = ray.O + ray.t * ray.D;
	float3 N = scene.GetNormal(ray.objIdx, I, ray.D);
	
	float r = rand() * (1.0 / RAND_MAX);
	float P = 0.8;
	if (r > P) return float3(0);
	Ray randomRay = Ray(I, random_in_uint_sphere());

	float3 color = float3(0);
	float cosine = abs(dot(-ray.D, N));

	r = rand() * (1.0 / RAND_MAX);
	if (r < mat.specularRate) 
	{
		float3 reflectRayDir = normalize(reflect(ray.D, N));
		randomRay.D = mix(reflectRayDir, randomRay.D, mat.roughness);
		color = PathTrace(randomRay, iter + 1) * cosine;
	}
	else if (mat.specularRate <= r && r <= mat.refractRate) 
	{
		float3 refractRayDir = normalize(refract(ray.D, N, float(mat.refractAngle)));
		randomRay.D = mix(refractRayDir, -randomRay.D, mat.refractRoughness);
		color = PathTrace(randomRay, iter + 1) * cosine;
	}
	else 
	{
		float3 srcColor = mat.color;
		float3 ptColor = PathTrace(randomRay, iter + 1) * cosine;
		color = ptColor * srcColor;   
	}

	return 1.25 * BRIGHTNESS * color;// / P;
}

// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------
void Renderer::Tick( float deltaTime )
{
	// animation
	static float animTime = 0;
	scene.SetTime( animTime += deltaTime * 0.002f );
	// pixel loop
	Timer t;
//real-time sampling
	printf("sample: %f\n", sample);
	// lines are executed as OpenMP parallel tasks (disabled in DEBUG)
#	pragma omp parallel for schedule(dynamic)
	for (int y = 0; y < SCRHEIGHT; y++)
	{
		// trace a primary ray for each pixel on the line
		for (int x = 0; x < SCRWIDTH; x++)
		{
			float3 color = PathTrace(camera.GetPrimaryRay(x, y));

			accumulator[x + y * SCRWIDTH] *= (sample - 1) / sample;
			accumulator[x + y * SCRWIDTH] += float4(color * (1 / sample), 0);
		}

		// translate accumulator contents to rgb32 pixels
		for (int dest = y * SCRWIDTH, x = 0; x < SCRWIDTH; x++)
			screen->pixels[dest + x] =
			RGBF32_to_RGB8(&accumulator[x + y * SCRWIDTH]);
	}
	sample++;

	// in game control
	if (GetKeyState('A') < 0) camera.Translate(float3(0.1, 0, 0));
	if (GetKeyState('D') < 0) camera.Translate(float3(-0.1, 0, 0));
	if (GetKeyState('S') < 0) camera.Translate(float3(0, 0, 0.1));
	if (GetKeyState('W') < 0) camera.Translate(float3(0, 0, -0.1));
	if (GetKeyState('Q') < 0) camera.Translate(float3(0, -0.1, 0));
	if (GetKeyState('E') < 0) camera.Translate(float3(0, 0.1, 0));
	if (GetKeyState(37) < 0) camera.Rotate(0, 0.05);
	if (GetKeyState(38) < 0) camera.Rotate(0.05, 0);
	if (GetKeyState(39) < 0) camera.Rotate(0, -0.05);
	if (GetKeyState(40) < 0) camera.Rotate(-0.05, 0);

	// performance report - running average - ms, MRays/s
	static float avg = 10, alpha = 1;
	avg = (1 - alpha) * avg + alpha * t.elapsed() * 1000;
	if (alpha > 0.05f) alpha *= 0.5f;
	float fps = 1000 / avg, rps = (SCRWIDTH * SCRHEIGHT) * fps;
	//printf( "%5.2fms (%.1fps) - %.1fMrays/s\n", avg, fps, rps / 1000000 );
}