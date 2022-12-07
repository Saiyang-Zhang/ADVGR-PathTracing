#pragma once

namespace Tmpl8
{

class Renderer : public TheApp
{
	const double BRIGHTNESS = 2.0f * 3.1415926f;

public:
	// game flow methods
	void Init();
	float3 Trace( Ray& ray, int sample );
	float3 PathTrace(Ray& ray, int iter);

	void Tick( float deltaTime );
	void Shutdown() { /* implement if you want to do something on exit */ }
	// input handling
	void MouseUp( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseDown( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseMove( int x, int y ) { mousePos.x = x, mousePos.y = y; }
	void MouseWheel( float y ) { /* implement if you want to handle the mouse wheel */ }
	void KeyUp( int key ) { /* implement if you want to handle keys */ }
	void KeyDown( int key ) { /* implement if you want to handle keys */ }
	// data members
	int2 mousePos;
	float4* accumulator;
	Scene scene;
	Camera camera;
	float sample;
};

} // namespace Tmpl8