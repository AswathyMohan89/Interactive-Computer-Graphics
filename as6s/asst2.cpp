////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////
//	These skeleton codes are later altered by Ming Jin,
//	for "CS6533: Interactive Computer Graphics", 
//	taught by Prof. Andy Nealen at NYU-Poly
////////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#if __GNUG__
#   include <tr1/memory>
#endif

#include <GL/glew.h>
#ifdef __MAC__
#   include <GLUT/glut.h>
#else
#   include <GL/glut.h>
#endif

#include "cvec.h"
#include "matrix4.h"
#include "geometrymaker.h"
#include "ppm.h"
#include "glsupport.h"
// asst3
#include "arcball.h"
#include "rigtform.h"
#include "quat.h"
// asst4
#include "asstcommon.h"
#include "scenegraph.h"
#include "drawer.h"
#include "picker.h"
// asst5
#include "sgutils.h"
#include <fstream>
// asst6
#include "geometry.h"

using namespace std;      // for string, vector, iostream, and other standard C++ stuff
using namespace tr1; // for shared_ptr

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.3.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.3. Make sure that your machine supports the version of GLSL you
// are using. In particular, on Mac OS X currently there is no way of using
// OpenGL 3.x with GLSL 1.3 when GLUT is used.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
const bool g_Gl2Compatible = false;


static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundY = -2.0;      // y coordinate of the ground
static const float g_groundSize = 10.0;   // half the ground length

static int g_windowWidth = 512;
static int g_windowHeight = 512;
static bool g_mouseClickDown = false;    // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static int g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;
// ========================================
// TODO: you can add global variables here
// ========================================
// CHANGED: Declared globals
// ========================================
// asst2
static int g_objIndex = 0; // cube 1
static int g_viewIndex = 0; // eyeView
static int g_skyWorld = 1; // currently sky-sky
// asst3
static double g_arcRadius = 1;		// initial ArcBall radius
static double g_arcScale = 1;	// initial ArcBall Scale
// asst4
static bool g_picking = false;
// asst5
static int g_currentFrame = -1;
static bool g_isAnimating = false;
static bool g_doSkipAnimation = false;
static int g_msBetweenKeyFrames = 2000;
static int g_framesPerSecond = 60;
static const string g_filename = "animation.kfs";

enum Mode{AB_Sky, AB_Picked, AB_No};
static Mode g_arcBallPos = AB_Sky;

static shared_ptr<Material> g_redDiffuseMat,
                            g_blueDiffuseMat,
                            g_bumpFloorMat,
                            g_arcballMat,
                            g_pickingMat,
                            g_lightMat;

shared_ptr<Material> g_overridingMaterial;

// --------- Geometry

typedef SgGeometryShapeNode MyShapeNode;

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;

// --------- Scene

static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, 3.0, -5.0);  // define two lights positions in world space
static RigTForm g_arcBall = RigTForm(Cvec3(0,0,0));

static Cvec3f g_objectColors[2] = {
	Cvec3f(1, 0, 0),
	Cvec3f(0, 0, 1)
};

static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot1Node, g_robot2Node;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode;

static shared_ptr<SgRbtNode> g_eyeNode, g_curObjNode;

static vector<vector<RigTForm>> g_keyFrames;

static shared_ptr<Texture> g_texture;
static shared_ptr<SgRbtNode> g_light1Node, g_light2Node;

///////////////// END OF G L O B A L S //////////////////////////////////////////////////


static void initGround() {
  int ibLen, vbLen;
  getPlaneVbIbLen(vbLen, ibLen);

  // Temporary storage for cube Geometry
  vector<VertexPNTBX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makePlane(g_groundSize*2, vtx.begin(), idx.begin());
  g_ground.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initCubes() {
  int ibLen, vbLen;
  getCubeVbIbLen(vbLen, ibLen);

  // Temporary storage for cube Geometry
  vector<VertexPNTBX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makeCube(1, vtx.begin(), idx.begin());
  g_cube.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initSphere() {
  int ibLen, vbLen;
  getSphereVbIbLen(20, 10, vbLen, ibLen);

  // Temporary storage for sphere Geometry
  vector<VertexPNTBX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);
  makeSphere(g_arcRadius * g_arcScale, 20, 10, vtx.begin(), idx.begin());
  g_sphere.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vtx.size(), idx.size()));
}

// takes a projection matrix and send to the the shaders
inline void sendProjectionMatrix(Uniforms& uniforms, const Matrix4& projMatrix) {
  uniforms.put("uProjMatrix", projMatrix);
}
// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
	if (g_windowWidth >= g_windowHeight)
		g_frustFovY = g_frustMinFov;
	else {
		const double RAD_PER_DEG = 0.5 * CS175_PI/180;
		g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) / RAD_PER_DEG;
	}
}

static Matrix4 makeProjectionMatrix() {
	return Matrix4::makeProjection(
		g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),
		g_frustNear, g_frustFar);
}

static void updateScale()
{
	double 
		x1 = rigTFormToMatrix(g_arcBall)(0,3),
		y1 = rigTFormToMatrix(g_arcBall)(1,3),
		z1 = rigTFormToMatrix(g_curObjNode->getRbt())(2,3) - rigTFormToMatrix(g_arcBall)(2,3);

	double dist = sqrt(x1 * x1 + y1 * y1 + z1 * z1);
	g_arcScale = getScreenToEyeScale(-dist, g_frustFovY, g_windowHeight) * g_windowHeight * .20;
	if(g_arcBallPos == AB_No) g_arcScale = 0;
}

static void drawStuff(bool picking) {
	initSphere();

	Uniforms uniforms;

	// build & send proj. matrix to vshader
	const Matrix4 projmat = makeProjectionMatrix();
	sendProjectionMatrix(uniforms, projmat);

	// use the skyRbt as the eyeRbt
	// ===============================
	// CHANGED: Use current object as eyeRbt
	// ===============================
	const RigTForm eyeRbt = getPathAccumRbt(g_world, g_curObjNode);
	const RigTForm invEyeRbt = inv(eyeRbt);

	const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4((getPathAccumRbt(g_world, g_light1Node)).getTranslation(), 1)); // g_light1 position in eye coordinates
	const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4((getPathAccumRbt(g_world, g_light2Node)).getTranslation(), 1)); // g_light2 position in eye coordinates
	uniforms.put("uLight", eyeLight1);
	uniforms.put("uLight2", eyeLight2);

	if (!picking) {
		Drawer drawer(invEyeRbt, uniforms);
		g_world->accept(drawer);

		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		Matrix4 MVM = rigTFormToMatrix(invEyeRbt * g_arcBall);
		Matrix4 NMVM = normalMatrix(MVM);
		sendModelViewNormalMatrix(uniforms, MVM, NMVM);
		uniforms.put("h_uColor", Cvec3(0, .8, 0));
		g_arcballMat->draw(*g_sphere, uniforms);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	else {
		Picker picker(invEyeRbt, uniforms);
		g_overridingMaterial = g_pickingMat;
		g_world->accept(picker);
		g_overridingMaterial.reset();
		glFlush();
		g_currentPickedRbtNode = picker.getRbtNodeAtXY(g_mouseClickX, g_mouseClickY);
		if (g_currentPickedRbtNode == g_groundNode)
			g_currentPickedRbtNode = shared_ptr<SgRbtNode>();   // set to NULL
		if(g_currentPickedRbtNode == NULL) {
			cout<<"Selected Self"<<endl;
			g_currentPickedRbtNode = g_curObjNode; // if nothing selected select self i.e currently manipulating object
			if(g_viewIndex != 0) {
				g_arcBall = getPathAccumRbt(g_world, g_currentPickedRbtNode);
				g_arcBallPos = AB_No;
			}
			else {
				g_skyWorld = 1;
				g_arcBall = g_world->getRbt();
				g_arcBallPos = AB_Sky;
			}
		}
		else {
			g_arcBall = getPathAccumRbt(g_world, g_currentPickedRbtNode);
			g_arcBallPos = AB_Picked;
		}
		updateScale();
	}
}

static void display() {	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // clear framebuffer color&depth

	drawStuff(false);

	glutSwapBuffers();                                    // show the back buffer (where we rendered stuff)

	checkGlErrors();
}

static void reshape(const int w, const int h) {
	g_windowWidth = w;
	g_windowHeight = h;
	glViewport(0, 0, w, h);
	cerr << "Size of window is now " << w << "x" << h << endl;
	updateFrustFovY();
	updateScale();
	glutPostRedisplay();
}

static void motion(const int x, const int y) {	
	const int mx = x, my = g_windowHeight - y - 1;
	const double dx = x - g_mouseClickX;
	const double dy = g_windowHeight - y - 1 - g_mouseClickY;

	RigTForm m;
	if (g_mouseLClickButton && !g_mouseRClickButton) {		// left button down?
		if(g_currentPickedRbtNode == g_curObjNode && g_skyWorld == 0 && g_curObjNode == g_eyeNode 
			|| g_curObjNode != g_eyeNode && g_currentPickedRbtNode == g_curObjNode
			|| g_skyWorld == 0 && g_objIndex == 0){
				// ego motion of cubes with respect themselves
				m.setRotation(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx));
		}
		else
		{
			Matrix4 ot = rigTFormToMatrix(inv(getPathAccumRbt(g_world, g_curObjNode)) * g_arcBall);
			Cvec2 cxy = getScreenSpaceCoord(Cvec3(ot(0,3), ot(1,3), ot(2,3)), makeProjectionMatrix(), g_frustNear, g_frustFovY, g_windowWidth, g_windowHeight);
			// get the boundry distance of arc ball;
			double bound = getScreenSpaceCoord(Cvec3(ot(0,3) + g_arcRadius * g_arcScale, ot(1,3), ot(2,3)), makeProjectionMatrix(), g_frustNear, g_frustFovY, g_windowWidth, g_windowHeight)[0] - cxy[0];

			// Screen Vectors
			Cvec2 p1 = Cvec2(g_mouseClickX, g_mouseClickY) - cxy;
			Cvec2 p2 = Cvec2(mx, my) - cxy;


			// Clamp to boundary
			double dist, z1, z2;
			dist = norm2(p1);
			if(dist > bound * bound) z1 = 0;
			else z1 = sqrt(bound * bound - dist);
			dist = norm2(p2);
			if(dist > bound * bound) z2 = 0;
			else z2 = sqrt(bound * bound - dist);

			//Get 3D Vectors
			Cvec3 v0 = normalize(Cvec3(p1, z1));
			Cvec3 v1 = normalize(Cvec3(p2, z2));

			m = inv(RigTForm(Quat(0, v1) * Quat(0, -v0[0], -v0[1], -v0[2])));			
		}
	}
	else if (g_mouseRClickButton && !g_mouseLClickButton) { // right button down?
		m = RigTForm(Cvec3(dx, dy, 0) * 0.01);
	}
	else if (g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton)) {  // middle or (left and right) button down?
		m = RigTForm(Cvec3(0, 0, -dy) * 0.01);		
	}

	if (g_mouseClickDown) {		
		if(g_arcBallPos == AB_Sky) {
			if(g_skyWorld == 0 && g_mouseLClickButton && !g_mouseRClickButton) m = inv(m); // invert only ego rotation
			else if(g_skyWorld == 1 && (g_mouseRClickButton)) m = inv(m); // else invert back;
			RigTForm wrt = g_skyWorld == 0 ? g_eyeNode->getRbt() : g_world->getRbt();
			m = g_mouseLClickButton && g_mouseRClickButton ? inv(m) : m;
			RigTForm A = transFact(wrt) * linFact(g_eyeNode->getRbt());
			g_eyeNode->setRbt( A * m * inv(A) * g_eyeNode->getRbt());
			g_arcBall = g_skyWorld == 0 ? g_eyeNode->getRbt() : g_world->getRbt();			
		}
		else {
			if(g_mouseLClickButton && !g_mouseRClickButton) m = inv(m); // invert only ego rotation
			RigTForm t = g_currentPickedRbtNode->getRbt();
			RigTForm A = transFact(t) * linFact(getPathAccumRbt(g_world, g_curObjNode));	// transform the any given part with respect to curObj View
			g_currentPickedRbtNode->setRbt(A * m * inv(A) * t);	// apply tranformation
			g_arcBall = getPathAccumRbt(g_world, g_currentPickedRbtNode);
		}
		g_curObjNode = g_viewIndex == 0 ? g_eyeNode : g_viewIndex == 1 ? g_robot1Node : g_robot2Node;	

		glutPostRedisplay(); // we always redraw if we changed the scene
	}

	g_mouseClickX = x;
	g_mouseClickY = g_windowHeight - y - 1;
}

static void constructRobot(shared_ptr<SgTransformNode>, shared_ptr<Material>);
void initScene();

static void reset()
{
	// =========================================================
	// TODO:
	// - reset g_skyRbt and g_objectRbt to their default values
	// - reset the views and manipulation mode to default
	// - reset sky camera mode to use the "world-sky" frame
	// =========================================================
	// CHANGED: set all to initial values
	// =========================================================	
	initScene();

	g_arcBall = RigTForm(Cvec3(0,0,0));

	g_currentFrame = -1;
	g_isAnimating = false;
	g_doSkipAnimation = false;
	g_msBetweenKeyFrames = 2000;
	g_framesPerSecond = 60;	

	for (size_t i = 0; i < g_keyFrames.size(); ++i)
		g_keyFrames[i].clear();
	g_keyFrames.clear();

	g_objIndex = 0;
	g_viewIndex = 0;
	g_skyWorld = 1;
	g_arcRadius = 1;
	g_arcScale = 1;
	g_arcBallPos = AB_Sky;	
}

static void pick() {
	// We need to set the clear color to black, for pick rendering.
	// so let's save the clear color
	GLdouble clearColor[4];
	glGetDoublev(GL_COLOR_CLEAR_VALUE, clearColor);

	glClearColor(0, 0, 0, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawStuff(true);

	// Uncomment below and comment out the glutPostRedisplay in mouse(...) call back
	// to see result of the pick rendering pass
	// glutSwapBuffers();

	//Now set back the clear color
	glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);

	checkGlErrors();
}

static void mouse(const int button, const int state, const int x, const int y) {
	g_mouseClickX = x;
	g_mouseClickY = g_windowHeight - y - 1;  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system

	g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
	g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
	g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

	g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
	g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
	g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

	g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;

	if (g_picking && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		pick();
		g_picking = false;
		g_arcBallPos = AB_Picked;
		cerr << "Picking mode is off" << endl;
		glutPostRedisplay(); // request redisplay since the arcball will have moved
	}

	if(!g_mouseClickDown) {
		updateScale();
		glutPostRedisplay();		
	}
}

static void toFrame(int f) {
	if(f == -1) return;
	vector<RigTForm> frame = g_keyFrames[f];
	vector<shared_ptr<SgRbtNode>> nodes;
	dumpSgRbtNodes(g_world, nodes);
	vector<shared_ptr<SgRbtNode>>::iterator niter = nodes.begin();

	for(size_t i = 0; i < nodes.size(); ++i, ++niter )
		(*niter)->setRbt(frame[i]);

	glutPostRedisplay();
}

static void saveFrame() {
	if(g_currentFrame == -1) return;

	vector<shared_ptr<SgRbtNode>> nodes;
	dumpSgRbtNodes(g_world, nodes);

	vector<RigTForm> rtfs;
	for(size_t i = 0; i< nodes.size(); ++i)
		rtfs.push_back(nodes[i]->getRbt());

	if(g_currentFrame == g_keyFrames.size())
	{
		g_keyFrames.push_back(rtfs);
		g_currentFrame = g_keyFrames.size() - 1;
		cout<<"Frame Created"<<endl;
	}
	else
	{
		g_keyFrames.erase(g_keyFrames.begin() + g_currentFrame);
		g_keyFrames.insert(g_keyFrames.begin() + g_currentFrame, rtfs);
		cout<<"Frame Updated"<<endl;
	}

}

bool interpolateAndDisplay(float t) {
	if(floor(t) + 1 == g_keyFrames.size() - 2) return true; // finished animation
	int frame = floor(t) + 1;

	vector<shared_ptr<SgRbtNode>> nodes;
	dumpSgRbtNodes(g_world, nodes);

	vector<shared_ptr<SgRbtNode>>::iterator niter = nodes.begin();

	vector<vector<RigTForm>>::iterator iter = g_keyFrames.begin();
	advance(iter, floor(t) + 1);

	vector<RigTForm> rtf0 = *(iter - 1);
	vector<RigTForm> rtf1 = *iter;
	++iter;
	vector<RigTForm> rtf2 = *iter;
	++iter;
	vector<RigTForm> rtf3 = *iter;

	vector<RigTForm>::iterator 
		r0iter = rtf0.begin(),
		r1iter = rtf1.begin(),
		r2iter = rtf2.begin(),
		r3iter = rtf3.begin();

	for(;r0iter != rtf0.end(), r1iter != rtf1.end(), r2iter != rtf2.end(), r3iter != rtf3.end(); ++r0iter, ++r1iter, ++r2iter, ++r3iter)
	{
		(*niter)->setRbt(interpolateCatmullRom(*r0iter, *r1iter, *r2iter, *r3iter, t - floor(t)));
		++niter;
	}

	glutPostRedisplay();
	return false;
}

static void animateTimerCallback(int ms) {
	float t = (float)ms / (float)g_msBetweenKeyFrames;
	if(g_doSkipAnimation) t = g_keyFrames.size() - 3;
	bool finished = interpolateAndDisplay(t);
	if(!finished) glutTimerFunc( 1000 / g_framesPerSecond, animateTimerCallback, ms + 1000 / g_framesPerSecond);
	else {
		cout<<"Finished Animation"<<endl;
		g_isAnimating = false;
		g_doSkipAnimation = false;
		toFrame(g_keyFrames.size() - 2);
		g_currentFrame = g_keyFrames.size() - 2;
	}
}

static void readKeyFrames() {
	ifstream fi(g_filename);
	if (!fi.is_open())
	{
		cout<<"Couldn't read the file!"<<endl;
		return;
	}


	g_keyFrames.clear();
	vector<RigTForm> rtf;

	string line;
	while (fi.good())
	{
		getline(fi, line);
		double t1, t2, t3, r1, r2, r3, r4;
		if (sscanf(line.c_str(), "%lf %lf %lf %lf %lf %lf %lf ", &t1, &t2, &t3, &r1, &r2, &r3, &r4) == 7)
		{
			Cvec3 t = Cvec3(t1, t2, t3);
			Quat r = Quat(r1, r2, r3, r4);
			RigTForm *tform = new RigTForm(t,r);
			rtf.push_back(*tform);
		}
		else
		{
			if (rtf.size() > 0)
				g_keyFrames.push_back(rtf);
			rtf.clear();
		}
	}   

	fi.close();
	cout<<"Succesfully read the keyframes"<<endl;
}

static void writeKeyFrames() {
	ofstream fo;
	fo.open(g_filename);
	if (!fo.is_open()) {
		cout<<"Couldn't read the file!"<<endl;
		return;
	}

	for (size_t i = 0; i < g_keyFrames.size(); i++)
	{
		vector<RigTForm> v = g_keyFrames[i];
		for (size_t j = 0; j < v.size(); j++)
		{
			Cvec3 t = v[j].getTranslation();
			fo<<t[0]<<" "<<t[1]<<" "<<t[2]<<" ";
			Quat r = v[j].getRotation();
			fo<<r[0]<<" " <<r[1]<<" "<< r[2]<<" "<<r[3]<<" "<<endl;
		}
		fo<<endl;
	}

	cout<<"Succesfully read the keyframes"<<endl;
	fo.close();
}

static void keyboard(const unsigned char key, const int x, const int y) {
	switch (key) {
	case 27:
		exit(0);                                  // ESC
	case 'h':
		cout << " ============== H E L P ==============\n\n"
			<< "h\t\thelp menu\n"
			<< "s\t\tsave screenshot\n"
			<< "f\t\tToggle flat shading on/off.\n"
			<< "o\t\tCycle object to edit\n"
			<< "v\t\tCycle view\n"
			<< "drag left mouse to rotate\n" << endl;
		break;
	case 's':
		glFlush();
		writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
		break;
	case 'v':
		g_viewIndex = (g_viewIndex + 1) % 3;
		g_objIndex = g_viewIndex;
		g_arcBall = g_skyWorld == 0 ? g_eyeNode->getRbt() : g_world->getRbt();
		switch(g_viewIndex)
		{
		case 0: g_curObjNode = g_eyeNode; g_arcBallPos = AB_Sky; cout<<"Current View: Sky"<<endl;    break;  // Sky    as Current View
		case 1: g_curObjNode = g_robot1Node; g_currentPickedRbtNode = g_robot1Node; g_arcBallPos = AB_Picked; cout<<"Current View: Cube 1"<<endl; break;  // Cube 1 as Current View
		case 2: g_curObjNode = g_robot2Node; g_currentPickedRbtNode = g_robot2Node; g_arcBallPos = AB_Picked; cout<<"Current View: Cube 2"<<endl; break;  // Cube 2 as Current View
		}
		break;
	case 'p':
		g_picking = !g_picking;
		g_skyWorld = 1;
		cout<<"Picking is "<<(g_picking ? "on" : "off")<<endl;
		break;
	case 'r':
		reset();		
		drawStuff(false);
		glutPostRedisplay();
		reset();
		cout << "reset objects and modes to defaults" << endl;
		break;
	case 'm':
		g_skyWorld = (g_skyWorld + 1) % 2;
		switch (g_skyWorld)
		{
		case 0: 
			cout<<"Current Frame: Sky-Sky"<<endl;
			g_arcBall = g_objIndex == 0 ? g_eyeNode->getRbt() : g_objIndex == 1 ? g_robot1Node->getRbt() : g_robot2Node->getRbt();
			break;  // Frame as Sky-Sky
		case 1: 
			cout<<"Current Frame: Sky-World"<<endl;
			g_arcBall = g_objIndex == 0 ? g_world->getRbt() : g_objIndex == 1 ? g_robot1Node->getRbt() : g_robot2Node->getRbt();
			break;  // Frame as Sky-World
		}
		break;
	case 'a':
		if(g_arcBallPos == AB_No) {
			g_arcBallPos = AB_Picked;			
		} else {
			g_arcBallPos = AB_No;
		}
		break;
		//animation controls
	case ' ':
		if(g_currentFrame > -1) toFrame(g_currentFrame);
		break;
	case 'u':
		saveFrame();
		break;
	case ',':	// < without shift
	case '<':	// < with shift
		if(g_currentFrame > 0) {
			toFrame(--g_currentFrame);
			cout<<"Stepped Backward"<<endl;
		}
		break;
	case '.':	// > without shift
	case '>':	// > with shift
		if(g_currentFrame >= 0 && g_currentFrame < (int)g_keyFrames.size() - 1) {
			toFrame(++g_currentFrame);
			cout<<"Stepped Forward"<<endl;
		}
		break;
	case 'd':
		// if first move to next
		if( g_currentFrame == 0 )
		{
			g_keyFrames.erase(g_keyFrames.begin());
			cout<<"Deleted First Frame"<<endl;
		}
		// else remove and goto previous
		else if( g_currentFrame > 0 )
		{
			cout<<"Deleting Frame "<<g_currentFrame<<endl;
			g_keyFrames.erase(g_keyFrames.begin() + g_currentFrame);
			--g_currentFrame;
			cout<<"Now at Frame "<<g_currentFrame;
		}

		// no frames left
		if(g_keyFrames.size() == 0) g_currentFrame = -1;

		toFrame(g_currentFrame);
		break;
	case 'n':
		g_currentFrame = g_keyFrames.size();
		saveFrame();
		break;
	case 'y':
		if(g_isAnimating)
			g_doSkipAnimation = true;
		else {
			if(g_keyFrames.size() < 4) cout<<"Sorry! Cannot play animation with less than 4 keyframes"<<endl;
			else
			{
				cout<<"Playing Animation"<<endl;
				g_isAnimating = true;
				animateTimerCallback(0);
			}
		}
		break;
	case '=': // without shift
	case '+': // with shift
		g_msBetweenKeyFrames -= 100;
		if(g_msBetweenKeyFrames < 100) g_msBetweenKeyFrames = 100;
		cout<<"Reduced Time Between from frames: "<<(float)g_msBetweenKeyFrames / 1000<<" seconds"<<endl;
		break;
	case '-': // without shift
	case '_': // with shift // not needed but if pressed?
		g_msBetweenKeyFrames += 100;
		if(g_msBetweenKeyFrames > 4000) g_msBetweenKeyFrames = 4000; // limited to 4000
		cout<<"Increased Time Between from frames: "<<(float)g_msBetweenKeyFrames / 1000<<" seconds"<<endl;
		break;
	case 'i': readKeyFrames(); break;
	case 'w': writeKeyFrames(); break;
	}	
	updateScale();
	glutPostRedisplay();
}

static void initGlutState(int argc, char * argv[]) {
	glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
	glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);  //  RGBA pixel channels and double buffering
	glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
	glutCreateWindow("Assignment 2");                       // title the window

	glutDisplayFunc(display);                               // display rendering callback
	glutReshapeFunc(reshape);                               // window reshape callback
	glutMotionFunc(motion);                                 // mouse movement callback
	glutMouseFunc(mouse);                                   // mouse click callback
	glutKeyboardFunc(keyboard);
}

static void initGLState() {
	glClearColor(128./255., 200./255., 255./255., 0.);
	glClearDepth(0.);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_GREATER);
	glReadBuffer(GL_BACK);
	if (!g_Gl2Compatible)
		glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initMaterials() {
  // Create some prototype materials
  Material diffuse("./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader");
  Material solid("./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader");

  // copy diffuse prototype and set red color
  g_redDiffuseMat.reset(new Material(diffuse));
  g_redDiffuseMat->getUniforms().put("uColor", Cvec3f(1, 0, 0));

  // copy diffuse prototype and set blue color
  g_blueDiffuseMat.reset(new Material(diffuse));
  g_blueDiffuseMat->getUniforms().put("uColor", Cvec3f(0, 0, 1));

  // normal mapping material
  g_bumpFloorMat.reset(new Material("./shaders/normal-gl3.vshader", "./shaders/normal-gl3.fshader"));
  g_bumpFloorMat->getUniforms().put("uTexColor", shared_ptr<ImageTexture>(new ImageTexture("Fieldstone.ppm", true)));
  g_bumpFloorMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("FieldstoneNormal.ppm", false)));

  // copy solid prototype, and set to wireframed rendering
  g_arcballMat.reset(new Material(solid));
  g_arcballMat->getUniforms().put("uColor", Cvec3f(0.27f, 0.82f, 0.35f));
  g_arcballMat->getRenderStates().polygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // copy solid prototype, and set to color white
  g_lightMat.reset(new Material(solid));
  g_lightMat->getUniforms().put("uColor", Cvec3f(1, 1, 1));

  // pick shader
  g_pickingMat.reset(new Material("./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader"));
};

static void initGeometry() {
	initGround();
	initCubes();
	initSphere();
}

static void constructRobot(shared_ptr<SgTransformNode> base, shared_ptr<Material> material) {

	const double 
		ARM_LEN = 0.7,
		ARM_THICK = 0.25,
		TORSO_LEN = 1.5,
		TORSO_THICK = 0.25,
		TORSO_WIDTH = 1,
		HEAD_RADIUS = .4;
	const int 
		NUM_JOINTS = 10,
		NUM_SHAPES = 10;

	struct JointDesc {
		int parent;
		float x, y, z;
	};

	JointDesc jointDesc[NUM_JOINTS] = {
		{-1}, // torso
		{0, TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper right arm
		{1, ARM_LEN, 0, 0},// lower right arm
		{0, -TORSO_WIDTH/2, TORSO_LEN/2, 0},
		{3, -ARM_LEN, 0, 0},
		{0, 0, TORSO_LEN/2, 0},
		{0, TORSO_WIDTH/2-ARM_THICK/2, -TORSO_LEN/2, 0},
		{6, 0, -ARM_LEN, 0},
		{0, -(TORSO_WIDTH/2-ARM_THICK/2), -TORSO_LEN/2, 0},
		{8, 0, -ARM_LEN, 0}
	};

	struct ShapeDesc {
		int parentJointId;
		float x, y, z, sx, sy, sz;
		shared_ptr<Geometry> geometry;
	};

	ShapeDesc shapeDesc[NUM_SHAPES] = {
		{0, 0,         0, 0, TORSO_WIDTH, TORSO_LEN, TORSO_THICK, g_cube }, // torso
		{1, ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // upper right arm
		{2, ARM_LEN/2, 0, 0, ARM_LEN * .6, ARM_THICK * .6, ARM_THICK * .6, g_sphere},// lower right arm
		{3, -ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube},
		{4, -ARM_LEN/2, 0, 0, ARM_LEN * .6, ARM_THICK * .6, ARM_THICK * .6, g_sphere},
		{5, 0, HEAD_RADIUS, 0, HEAD_RADIUS, HEAD_RADIUS, HEAD_RADIUS, g_sphere},
		{6, 0, -ARM_LEN/2, 0, ARM_THICK, ARM_LEN, ARM_THICK, g_cube},
		{7, 0, -ARM_LEN/2, 0, ARM_THICK * .6, ARM_LEN *.6, ARM_THICK *.6, g_sphere},
		{8, 0, -ARM_LEN/2, 0, ARM_THICK, ARM_LEN, ARM_THICK, g_cube},
		{9, 0, -ARM_LEN/2, 0, ARM_THICK * .6, ARM_LEN *.6, ARM_THICK *.6, g_sphere}
	};

	shared_ptr<SgTransformNode> jointNodes[NUM_JOINTS];

	for (int i = 0; i < NUM_JOINTS; ++i) {
		if (jointDesc[i].parent == -1)
			jointNodes[i] = base;
		else {
			jointNodes[i].reset(new SgRbtNode(RigTForm(Cvec3(jointDesc[i].x, jointDesc[i].y, jointDesc[i].z))));
			jointNodes[jointDesc[i].parent]->addChild(jointNodes[i]);
		}
	}
	for (int i = 0; i < NUM_SHAPES; ++i) {
		shared_ptr<MyShapeNode> shape(
			new MyShapeNode(shapeDesc[i].geometry,
			material,
			Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
			Cvec3(0, 0, 0),
			Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
		jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
	}
}

static void initScene() {
	g_world.reset(new SgRootNode());

	g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.25, 4.0))));
	g_eyeNode = g_skyNode;
	g_curObjNode = g_eyeNode;

	g_groundNode.reset(new SgRbtNode());
	g_groundNode->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(0, g_groundY, 0))));
	// fixed
	// g_groundNode->addChild(shared_ptr<MyShapeNode>(
	//  	new MyShapeNode(g_ground, Cvec3(0.1, 0.95, 0.1))));

	// Add light to the scene

	g_light1Node.reset(new SgRbtNode(RigTForm(g_light1)));
    g_light1Node->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_sphere, g_lightMat)));
    g_light2Node.reset(new SgRbtNode(RigTForm(g_light2)));
    g_light2Node->addChild(shared_ptr<MyShapeNode>(new MyShapeNode(g_sphere, g_lightMat)));

	g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-2, 1, 0))));
	g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(2, 1, 0))));

	constructRobot(g_robot1Node, g_redDiffuseMat); // a Red robot
	constructRobot(g_robot2Node, g_blueDiffuseMat); // a Blue robot

	g_world->addChild(g_skyNode);
	g_world->addChild(g_groundNode);
	g_world->addChild(g_robot1Node);
	g_world->addChild(g_robot2Node);
	g_world->addChild(g_light1Node);
    g_world->addChild(g_light2Node);
}

int main(int argc, char * argv[]) {
	try {
		initGlutState(argc,argv);

		glewInit(); // load the OpenGL extensions

		cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0" : "Will use OpenGL 3.x / GLSL 1.3") << endl;
		if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
			throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.3");
		else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
			throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.0");

		initGLState();
		initMaterials();
		initGeometry();
		initScene();

		glutMainLoop();
		return 0;
	}
	catch (const runtime_error& e) {
		cout << "Exception caught: " << e.what() << endl;
		return -1;
	}
}
