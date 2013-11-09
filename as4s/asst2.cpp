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
static int g_objIndex = 0; // cube 1
static int g_viewIndex = 0; // eyeView
static int g_skyWorld = 1; // currently sky-sky

static double g_arcRadius = 1;		// initial ArcBall radius
static double g_arcScale = 1;	// initial ArcBall Scale

static bool g_picking = false;

enum Mode{AB_Sky, AB_Picked, AB_No};
static Mode g_arcBallPos = AB_Sky;

static int PICKING_SHADER = 2;
static const int g_numShaders = 3;
static const char * const g_shaderFiles[g_numShaders][2] = {
	{"./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader"},
	{"./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader"},
	{"./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader"}
};
static const char * const g_shaderFilesGl2[g_numShaders][2] = {
	{"./shaders/basic-gl2.vshader", "./shaders/diffuse-gl2.fshader"},
	{"./shaders/basic-gl2.vshader", "./shaders/solid-gl2.fshader"},
	{"./shaders/basic-gl2.vshader", "./shaders/pick-gl2.fshader"}
};
static vector<shared_ptr<ShaderState> > g_shaderStates; // our global shader states

// --------- Geometry

// Macro used to obtain relative offset of a field within a struct
#define FIELD_OFFSET(StructType, field) &(((StructType *)0)->field)

// A vertex with floating point position and normal
struct VertexPN {
	Cvec3f p, n;

	VertexPN() {}
	VertexPN(float x, float y, float z,
		float nx, float ny, float nz)
		: p(x,y,z), n(nx, ny, nz)
	{}

	// Define copy constructor and assignment operator from GenericVertex so we can
	// use make* functions from geometrymaker.h
	VertexPN(const GenericVertex& v) {
		*this = v;
	}

	VertexPN& operator = (const GenericVertex& v) {
		p = v.pos;
		n = v.normal;
		return *this;
	}
};

struct Geometry {
	GlBufferObject vbo, ibo;
	int vboLen, iboLen;

	Geometry(VertexPN *vtx, unsigned short *idx, int vboLen, int iboLen) {
		this->vboLen = vboLen;
		this->iboLen = iboLen;

		// Now create the VBO and IBO
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPN) * vboLen, vtx, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * iboLen, idx, GL_STATIC_DRAW);
	}

	void draw(const ShaderState& curSS) {
		// Enable the attributes used by our shader
		safe_glEnableVertexAttribArray(curSS.h_aPosition);
		safe_glEnableVertexAttribArray(curSS.h_aNormal);

		// bind vbo
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		safe_glVertexAttribPointer(curSS.h_aPosition, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, p));
		safe_glVertexAttribPointer(curSS.h_aNormal, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, n));

		// bind ibo
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

		// draw!
		glDrawElements(GL_TRIANGLES, iboLen, GL_UNSIGNED_SHORT, 0);

		// Disable the attributes used by our shader
		safe_glDisableVertexAttribArray(curSS.h_aPosition);
		safe_glDisableVertexAttribArray(curSS.h_aNormal);
	}
};

typedef SgGeometryShapeNode<Geometry> MyShapeNode;

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;

// --------- Scene

static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, -3.0, -5.0);  // define two lights positions in world space
static RigTForm g_arcBall = RigTForm(Cvec3(0,0,0));

static Cvec3f g_objectColors[2] = {
	Cvec3f(1, 0, 0),
	Cvec3f(0, 0, 1)
};

static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot1Node, g_robot2Node;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode;

static shared_ptr<SgRbtNode> g_eyeNode, g_curObjNode;

///////////////// END OF G L O B A L S //////////////////////////////////////////////////


static void initSphere() {
	int ibLen, vbLen;
	getSphereVbIbLen(20, 10, vbLen, ibLen);

	vector<VertexPN> vtx(vbLen);
	vector<unsigned short> idx(ibLen);

	makeSphere(g_arcRadius * g_arcScale, 20, 10, vtx.begin(), idx.begin());
	g_sphere.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initGround() {
	// A x-z plane at y = g_groundY of dimension [-g_groundSize, g_groundSize]^2
	VertexPN vtx[4] = {
		VertexPN(-g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
		VertexPN(-g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
		VertexPN( g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
		VertexPN( g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
	};
	unsigned short idx[] = {0, 1, 2, 0, 2, 3};
	g_ground.reset(new Geometry(&vtx[0], &idx[0], 4, 6));
}

static void initCubes() {
	int ibLen, vbLen;
	getCubeVbIbLen(vbLen, ibLen);

	// Temporary storage for cube geometry
	vector<VertexPN> vtx(vbLen);
	vector<unsigned short> idx(ibLen);

	makeCube(1, vtx.begin(), idx.begin());
	g_cube.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
}

// takes a projection matrix and send to the the shaders
static void sendProjectionMatrix(const ShaderState& curSS, const Matrix4& projMatrix) {
	GLfloat glmatrix[16];
	projMatrix.writeToColumnMajorMatrix(glmatrix); // send projection matrix
	safe_glUniformMatrix4fv(curSS.h_uProjMatrix, glmatrix);
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

static void drawStuff(const ShaderState& curSS, bool picking) {
	initSphere();

	// build & send proj. matrix to vshader
	const Matrix4 projmat = makeProjectionMatrix();
	sendProjectionMatrix(curSS, projmat);

	// use the skyRbt as the eyeRbt
	// ===============================
	// CHANGED: Use current object as eyeRbt
	// ===============================
	const RigTForm eyeRbt = getPathAccumRbt(g_world, g_curObjNode);
	const RigTForm invEyeRbt = inv(eyeRbt);

	const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4(g_light1, 1)); // g_light1 position in eye coordinates
	const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4(g_light2, 1)); // g_light2 position in eye coordinates
	safe_glUniform3f(curSS.h_uLight, eyeLight1[0], eyeLight1[1], eyeLight1[2]);
	safe_glUniform3f(curSS.h_uLight2, eyeLight2[0], eyeLight2[1], eyeLight2[2]);	

	if (!picking) {
		Drawer drawer(invEyeRbt, curSS);
		g_world->accept(drawer);

		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		Matrix4 MVM = rigTFormToMatrix(invEyeRbt * g_arcBall);
		Matrix4 NMVM = normalMatrix(MVM);
		sendModelViewNormalMatrix(curSS, MVM, NMVM);
		safe_glUniform3f(curSS.h_uColor, 0,.8,0);
		g_sphere->draw(curSS);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	else {
		Picker picker(invEyeRbt, curSS);
		g_world->accept(picker);
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
	glUseProgram(g_shaderStates[g_activeShader]->program);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // clear framebuffer color&depth

	drawStuff(*g_shaderStates[g_activeShader], false);

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

static void constructRobot(shared_ptr<SgTransformNode>, const Cvec3&);
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

	// using PICKING_SHADER as the shader
	glUseProgram(g_shaderStates[PICKING_SHADER]->program);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawStuff(*g_shaderStates[PICKING_SHADER], true);

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
		cout<<g_arcRadius * g_arcScale;
	}
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
	case 'f':
		g_activeShader ^= 1;
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
		drawStuff(*g_shaderStates[g_activeShader], false);
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

static void initShaders() {
	g_shaderStates.resize(g_numShaders);
	for (int i = 0; i < g_numShaders; ++i) {
		if (g_Gl2Compatible)
			g_shaderStates[i].reset(new ShaderState(g_shaderFilesGl2[i][0], g_shaderFilesGl2[i][1]));
		else
			g_shaderStates[i].reset(new ShaderState(g_shaderFiles[i][0], g_shaderFiles[i][1]));
	}
}

static void initGeometry() {
	initGround();
	initCubes();
	initSphere();
}

static void constructRobot(shared_ptr<SgTransformNode> base, const Cvec3& color) {

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
			color,
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
	g_groundNode->addChild(shared_ptr<MyShapeNode>(
		new MyShapeNode(g_ground, Cvec3(0.1, 0.95, 0.1))));

	g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-2, 1, 0))));
	g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(2, 1, 0))));

	constructRobot(g_robot1Node, Cvec3(1, 0, 0)); // a Red robot
	constructRobot(g_robot2Node, Cvec3(0, 0, 1)); // a Blue robot

	g_world->addChild(g_skyNode);
	g_world->addChild(g_groundNode);
	g_world->addChild(g_robot1Node);
	g_world->addChild(g_robot2Node);
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
		initShaders();
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
