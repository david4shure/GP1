////////////////////////////////////////////////////////////////////////
//
//   Westmont College
//   CS150 : 3D Computer Graphics
//   Professor David Hunter
//
//   Code derived from book code for _Foundations of 3D Computer Graphics_
//   by Steven Gortler.  See AUTHORS file for more details.
//
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

using namespace std; // for string, vector, iostream, and other standard C++ stuff
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
static const bool g_Gl2Compatible = true;

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

struct ShaderState {
  GlProgram program;

  // Handles to uniform variables
  GLint h_uProjMatrix;
  GLint h_uModelViewMatrix;
  GLint h_uColor;
  GLint h_uTexUnit0; 

  // Handles to vertex attributes
  GLint h_aPosition;
  GLint h_aTexCoord0;

  ShaderState(const char* vsfn, const char* fsfn) {
    readAndCompileShader(program, vsfn, fsfn);

    const GLuint h = program; // short hand

    // Retrieve handles to uniform variables
    h_uProjMatrix = safe_glGetUniformLocation(h, "uProjMatrix");
    h_uModelViewMatrix = safe_glGetUniformLocation(h, "uModelViewMatrix");
    h_uColor = safe_glGetUniformLocation(h, "uColor");
    h_uTexUnit0 = safe_glGetUniformLocation(h, "uTexUnit0"); 

    // Retrieve handles to vertex attributes
    h_aPosition = safe_glGetAttribLocation(h, "aPosition");
    h_aTexCoord0 = safe_glGetAttribLocation(h, "aTexCoord0"); 

    if (!g_Gl2Compatible)
      glBindFragDataLocation(h, 0, "fragColor");
    checkGlErrors();
  }

};

static const char * const g_shaderFiles[2] = 
  {"./shaders/basic-gl3.vshader", "./shaders/textured-gl3.fshader"};
static const char * const g_shaderFilesGl2[2] = 
  {"./shaders/basic-gl2.vshader", "./shaders/textured-gl2.fshader"};

static shared_ptr<ShaderState>  g_shaderState; // our global shader state

static shared_ptr<GlTexture> g_tex0, g_tex1; // our global texture instance

// --------- Geometry

// Macro used to obtain relative offset of a field within a struct
#define FIELD_OFFSET(StructType, field) &(((StructType *)0)->field)

// A vertex with floating point Position, Normal, and one set of teXture coordinates;
// We won't use normals for Project #2, but we will need them later.
struct VertexPNX {
  Cvec3f p, n; // position and normal vectors
  Cvec2f x; // texture coordinates

  VertexPNX() {}

  VertexPNX(float x, float y, float z,
            float nx, float ny, float nz,
            float u, float v)
    : p(x,y,z), n(nx, ny, nz), x(u, v) 
  {}

  VertexPNX(const Cvec3f& pos, const Cvec3f& normal, const Cvec2f& texCoords)
    :  p(pos), n(normal), x(texCoords) {}

  VertexPNX(const Cvec3& pos, const Cvec3& normal, const Cvec2& texCoords)
    : p(pos[0], pos[1], pos[2]), n(normal[0], normal[1], normal[2]), x(texCoords[0], texCoords[1]) {}

  // Define copy constructor and assignment operator from GenericVertex so we can
  // use make* functions from geometrymaker.h
  VertexPNX(const GenericVertex& v) {
    *this = v;
  }

  VertexPNX& operator = (const GenericVertex& v) {
    p = v.pos;
    n = v.normal;
    x = v.tex;
    return *this;
  }
};

struct Geometry {
  GlBufferObject vbo, ibo;
  int vboLen, iboLen;

  Geometry(VertexPNX *vtx, unsigned short *idx, int vboLen, int iboLen) {
    this->vboLen = vboLen;
    this->iboLen = iboLen;

    // Now create the VBO and IBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPNX) * vboLen, vtx, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * iboLen, idx, GL_STATIC_DRAW);
  }

  void draw(const ShaderState& curSS) {
    // Enable the attributes used by our shader
    safe_glEnableVertexAttribArray(curSS.h_aPosition);
    safe_glEnableVertexAttribArray(curSS.h_aTexCoord0); 

    // bind vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    safe_glVertexAttribPointer(curSS.h_aPosition, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPNX), FIELD_OFFSET(VertexPNX, p));
    safe_glVertexAttribPointer(curSS.h_aTexCoord0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexPNX), FIELD_OFFSET(VertexPNX, x));

    // bind ibo
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    // draw!
    glDrawElements(GL_TRIANGLES, iboLen, GL_UNSIGNED_SHORT, 0);

    // Disable the attributes used by our shader
    safe_glDisableVertexAttribArray(curSS.h_aPosition);
    safe_glDisableVertexAttribArray(curSS.h_aTexCoord0); 
  }
};

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube;

// --------- Scene

static Matrix4 g_skyRbt = Matrix4::makeTranslation(Cvec3(0.0, 0.25, 4.0));
static Matrix4 g_objectRbt[1] = {Matrix4::makeTranslation(Cvec3(0,0,0))};  // currently only 1 obj is defined
static Cvec3f g_objectColors[1] = {Cvec3f(1, 0, 0)};

///////////////// END OF G L O B A L S //////////////////////////////////////////////////

static void initGround() {
  // A x-z plane at y = g_groundY of dimension [-g_groundSize, g_groundSize]^2
  VertexPNX vtx[4] = {
    VertexPNX(-g_groundSize, g_groundY, -g_groundSize, 0, 1, 0, 0, 0), 
    VertexPNX(-g_groundSize, g_groundY,  g_groundSize, 0, 1, 0, 0, 1), 
    VertexPNX( g_groundSize, g_groundY,  g_groundSize, 0, 1, 0, 1, 1),
    VertexPNX( g_groundSize, g_groundY, -g_groundSize, 0, 1, 0, 1, 0),
  };
  unsigned short idx[] = {0, 1, 2, 0, 2, 3};
  g_ground.reset(new Geometry(&vtx[0], &idx[0], 4, 6));
}

static void initCubes() {
  int ibLen, vbLen;
  getCubeVbIbLen(vbLen, ibLen);

  // Temporary storage for cube geometry
  vector<VertexPNX> vtx(vbLen);
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

// takes model view matrix to the shaders
static void sendModelViewMatrix(const ShaderState& curSS, const Matrix4& MVM) {
  GLfloat glmatrix[16];
  MVM.writeToColumnMajorMatrix(glmatrix); // send MVM
  safe_glUniformMatrix4fv(curSS.h_uModelViewMatrix, glmatrix);
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

static void drawStuff() {
  // short hand for current shader state
  const ShaderState& curSS = *g_shaderState;

  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  sendProjectionMatrix(curSS, projmat);

  // use the skyRbt as the eyeRbt
  const Matrix4 eyeRbt = g_skyRbt;
  const Matrix4 invEyeRbt = inv(eyeRbt);

  // draw ground
  // ===========
  //
  const Matrix4 groundRbt = Matrix4();  // identity
  Matrix4 MVM = invEyeRbt * groundRbt;
  sendModelViewMatrix(curSS, MVM);
  safe_glUniform3f(curSS.h_uColor, 0.1, 0.95, 0.1); // set color
  safe_glUniform1i(curSS.h_uTexUnit0, 0); // texture unit 0 for ground
  g_ground->draw(curSS);

  // draw cubes
  // ==========
  MVM = invEyeRbt * g_objectRbt[0];
  sendModelViewMatrix(curSS, MVM);
  safe_glUniform3f(curSS.h_uColor, g_objectColors[0][0], g_objectColors[0][1], g_objectColors[0][2]);
  safe_glUniform1i(curSS.h_uTexUnit0, 1); // texture unit 1 for cube
  g_cube->draw(curSS);
}

static void display() {
  glUseProgram(g_shaderState->program);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // clear framebuffer color&depth

  drawStuff();

  glutSwapBuffers();                                    // show the back buffer (where we rendered stuff)

  checkGlErrors();
}

static void reshape(const int w, const int h) {
  g_windowWidth = w;
  g_windowHeight = h;
  glViewport(0, 0, w, h);
  cerr << "Size of window is now " << w << "x" << h << endl;
  updateFrustFovY();
  glutPostRedisplay();
}

static void motion(const int x, const int y) {
  const double dx = x - g_mouseClickX;
  const double dy = g_windowHeight - y - 1 - g_mouseClickY;

  Matrix4 m;
  if (g_mouseLClickButton && !g_mouseRClickButton) { // left button down?
    m = Matrix4::makeXRotation(-dy) * Matrix4::makeYRotation(dx);
  }
  else if (g_mouseRClickButton && !g_mouseLClickButton) { // right button down?
    m = Matrix4::makeTranslation(Cvec3(dx, dy, 0) * 0.01);
  }
  else if (g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton)) {  // middle or (left and right) button down?
    m = Matrix4::makeTranslation(Cvec3(0, 0, -dy) * 0.01);
  }

  if (g_mouseClickDown) {
    g_objectRbt[0] *= m; // Caddywumpus!  FIXME
    glutPostRedisplay(); // we always redraw if we changed the scene
  }

  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;
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
}


static void keyboard(const unsigned char key, const int x, const int y) {
  switch (key) {
  case 27:
    exit(0);                                  // ESC
  case 'h':
    cout << " ============== H E L P ==============\n\n"
    << "h\t\thelp menu\n"
    << "s\t\tsave screenshot\n"
    << "o\t\tCycle object to manipulate\n"
    << "v\t\tCycle view\n"
    << "m\t\tToggle Sky-Sky or World-Sky when manipulating sky\n"
    << "drag left mouse to rotate\n" 
    << "drag middle mouse to translate in/out \n" 
    << "drag right mouse to translate up/down/left/right\n" 
    << endl;
    break;
  case 's':
    glFlush();
    writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
    cout << "Screenshot written to out.ppm." << endl;
    break;
  }
  glutPostRedisplay();
}

static void initGlutState(int argc, char * argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);  //  RGBA pixel channels and double buffering
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("CS-150: Project 2");                  // title the window

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
    if (g_Gl2Compatible)
      g_shaderState.reset(new ShaderState(g_shaderFilesGl2[0], g_shaderFilesGl2[1]));
    else
      g_shaderState.reset(new ShaderState(g_shaderFiles[0], g_shaderFiles[1]));
}

static void initGeometry() {
  initGround();
  initCubes();
}

static void loadTexture(GLuint texHandle, const char *ppmFilename) {
  int texWidth, texHeight;
  vector<PackedPixel> pixData;

  ppmRead(ppmFilename, texWidth, texHeight, pixData);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texHandle);
  glTexImage2D(GL_TEXTURE_2D, 0, g_Gl2Compatible ? GL_RGB : GL_SRGB, texWidth, texHeight,
               0, GL_RGB, GL_UNSIGNED_BYTE, &pixData[0]);
  checkGlErrors();
}

static void initTextures() {
  g_tex0.reset(new GlTexture());
  g_tex1.reset(new GlTexture());

  loadTexture(*g_tex0, "checker256.ppm");
  loadTexture(*g_tex1, "roughbricks.ppm");

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, *g_tex0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, *g_tex1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
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
    initTextures();

    glutMainLoop();
    return 0;
  }
  catch (const runtime_error& e) {
    cout << "Exception caught: " << e.what() << endl;
    return -1;
  }
}
