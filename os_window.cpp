
#include "os_generic.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Opengl includes
//#include <GL/gl.h>
#include "include/glad.h"
#include "include/wglext.h"

// D3D11 includes
#include <d3d11.h>
#include <dxgi1_3.h>
#include <dxgidebug.h>
#include <d3d11sdklayers.h>

// Input
#include <Xinput.h>

#include <Windowsx.h>
#include <dwmapi.h>
#include <stdint.h>
#include <stdbool.h>

#include <shellscalingapi.h>

#include "base.h"

struct Win32_Message
{
    UINT kind;
    WPARAM wParam;
    LPARAM lParam;
};

struct OS_Context
{
    bool init;
    
    uint64_t qpcFreq;  // Used for query performance counter
    
    HWND window;
    HDC windowDC;
    
    void* mainFiber;
    void* eventsFiber;
    bool isInModalLoop;
    bool quit;
    
    // It's expected that the user won't
    // press buttons hundreds of times per frame
#define MaxBufferedInputs 20
    Win32_Message bufferedInputs[MaxBufferedInputs];
    int numInputs;
    
    // Currently used cursor. Can be nullptr which
    // means "hide the cursor"
    HCURSOR curCursor;
    
    // NOTE: Used when not in full screen mode
    bool usingDwm;
    
    OS_GraphicsLib usedGfxLib;
};

struct WGL_Context
{
    // Needed WGL function pointers
    PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB;
    PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB;
    PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;
    
    HGLRC glContext;
};

struct D3D11_Context
{
    IDXGISwapChain* swapchain;
    ID3D11Device* device;
    ID3D11DeviceContext* deviceContext;
};

static OS_Context win32;
static WGL_Context wgl;
static D3D11_Context d3d11;

// Stubs and signatures
// XInputGetState
#define XInputGetState_Signature(name) DWORD WINAPI name(DWORD dwUserIndex, XINPUT_STATE* pState)
typedef XInputGetState_Signature(XInputGetState_Type);
XInputGetState_Signature(XInputGetState_Stub)
{
    return 0;
}
static XInputGetState_Type* XInputGetState_ = XInputGetState_Stub;
#define XInputGetState XInputGetState_

// XInputSetState
#define XInputSetState_Signature(name) DWORD WINAPI name(DWORD dwUserIndex, XINPUT_VIBRATION* pVibration)
typedef XInputSetState_Signature(XInputSetState_Type);
XInputSetState_Signature(XInputSetState_Stub)
{
    return 0;
}
static XInputSetState_Type* XInputSetState_ = XInputSetState_Stub;
#define XInputSetState XInputSetState_

// Credits to mmozeiko
// https://gist.github.com/mmozeiko/ed2ad27f75edf9c26053ce332a1f6647
void Win32_GetWGLFunctions()
{
    // To get WGL functions, we first need a valid
    // OpenGL context. For that, we need to create
    // a dummy window for a dummy OpenGL context
    HWND dummy = CreateWindowEx(0, "STATIC", "Dummy", WS_OVERLAPPED,
                                CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
                                nullptr, nullptr, nullptr, nullptr);
    
    assert(dummy && "Failed to create dummy window");
    
    HDC dc = GetDC(dummy);
    assert(dc && "Failed to get device context for dummy window");
    
    PIXELFORMATDESCRIPTOR desc = {0};
    desc.nSize      = sizeof(desc);
    desc.nVersion   = 1;
    desc.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    desc.iPixelType = PFD_TYPE_RGBA;
    desc.cColorBits = 24;
    
    // NOTE: Whyyyyyy does this take ~1sec on my machine???
    // Seems to be a widespread driver problem that can't be fixed
    int format = ChoosePixelFormat(dc, &desc);
    if(!format)
        OS_FatalError("Cannot choose OpenGL pixel format for dummy window.");
    
    int success = DescribePixelFormat(dc, format, sizeof(desc), &desc);
    assert(success && "Failed to describe OpenGL pixel format");
    
    // Reason to create dummy window is that SetPixelFormat can be called only
    // once for this window
    if(!SetPixelFormat(dc, format, &desc))
        OS_FatalError("Cannot set OpenGL pixel format for dummy window.");
    
    HGLRC rc = wglCreateContext(dc);
    assert(rc && "Failed to create OpenGL context for dummy window.");
    
    success = wglMakeCurrent(dc, rc);
    assert(success && "Failed to make current OpenGL context for dummy window.");
    
    // https://www.khronos.org/registry/OpenGL/extensions/ARB/WGL_ARB_extensions_string.txt
    PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB = 
    (PFNWGLGETEXTENSIONSSTRINGARBPROC)wglGetProcAddress("wglGetExtensionsStringARB");
    
    if (!wglGetExtensionsStringARB)
        OS_FatalError("OpenGL does not support WGL_ARB_extensions_string extension!");
    
    const char* ext = wglGetExtensionsStringARB(dc);
    assert(ext && "Failed to get OpenGL WGL extension string");
    
    // Get the only three extensions we need to create the context
    const char* start = ext;
    while(true)
    {
        while(*ext != '\0' && *ext != ' ') ++ext;
        
        size_t length = ext - start;
        if(strncmp("WGL_ARB_pixel_format", start, length) == 0)
        {
            // https://www.khronos.org/registry/OpenGL/extensions/ARB/WGL_ARB_pixel_format.txt
            wgl.wglChoosePixelFormatARB = (PFNWGLCHOOSEPIXELFORMATARBPROC)wglGetProcAddress("wglChoosePixelFormatARB");
        }
        else if(strncmp("WGL_ARB_create_context", start, length) == 0)
        {
            // https://www.khronos.org/registry/OpenGL/extensions/ARB/WGL_ARB_create_context.txt
            wgl.wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
        }
        else if(strncmp("WGL_EXT_swap_control", start, length) == 0)
        {
            // https://www.khronos.org/registry/OpenGL/extensions/EXT/WGL_EXT_swap_control.txt
            wgl.wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
        }
        
        if(*ext == '\0') break;
        
        ++ext;
        start = ext;
    }
    
    if(!wgl.wglChoosePixelFormatARB ||
       !wgl.wglCreateContextAttribsARB ||
       !wgl.wglSwapIntervalEXT)
    {
        OS_FatalError("OpenGL does not support required WGL extensions for modern context.");
    }
    
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(rc);
    ReleaseDC(dummy, dc);
    DestroyWindow(dummy);
}

static bool Win32_LoadXInput()
{
    bool ok = true;
    HMODULE lib = LoadLibrary("xinput1_3.dll");
    if(lib)
    {
        auto setStateProc = GetProcAddress(lib, "XInputSetState");
        auto getStateProc = GetProcAddress(lib, "XInputGetState");
        
        if(setStateProc && getStateProc)
        {
            XInputGetState = (XInputGetState_Type*)getStateProc;
            XInputSetState = (XInputSetState_Type*)setStateProc;
        }
        else
            ok = false;
    }
    else
        ok = false;
    
    return ok;
}

#ifndef NDEBUG
static void APIENTRY OpenGLDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                                         GLsizei length, const GLchar* message, const void* user)
{
    OS_DebugMessage(message);
    OS_DebugMessage("\n");
    
    if(type == GL_DEBUG_TYPE_ERROR && severity == GL_DEBUG_SEVERITY_HIGH)
    {
        if(OS_IsDebuggerPresent())
        {
            assert(!"OpenGL error - check the callstack in debugger");
        }
        OS_FatalError("OpenGL API usage error! Use debugger to examine call stack!");
    }
}
#endif

void Win32_BreakOutOfModalLoop()
{
    win32.isInModalLoop = true;
    SwitchToFiber(win32.mainFiber);
    win32.isInModalLoop = false;
}

void Win32_BufferInputMessage(UINT message, WPARAM wParam, LPARAM lParam)
{
    // Save these input events until the next call to
    // OS_PollInput
    auto& curInput = win32.bufferedInputs[win32.numInputs % MaxBufferedInputs];
    curInput = { message, wParam, lParam };
    
    ++win32.numInputs;
    if(win32.numInputs > MaxBufferedInputs)
        win32.numInputs = MaxBufferedInputs;
}

// NOTE: Window callback function. Since Microsoft wants programs to get stuck
// inside this function at all costs, a weird hack is being used to "break free"
// of this function. Using a fiber I go back to regular code execution during
// modal loop events (which are events that don't let the program exit this)
// In addition to this nonsense, at some points when stuck in a modal loop, no
// messages are sent so there's no way to execute any code, so to solve that
// we need to fire "timer" messages which have a very low precision of 10ms.
// This will lead to stuttering during the modal loops (there is no way to fix it).
// This is insane but it's also the only -somewhat- clean way to keep rendering
// stuff while in the modal loop. (using a separate thread also has issues)
LRESULT CALLBACK WndProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
{
    struct DeferMessage
    {
        bool active;  // If this is true it means that we have a message saved.
        UINT msg;
        WPARAM wParam;
        LPARAM lParam;
    } typedef DeferMessage;
    
    static DeferMessage captionClick   = {0};
    static DeferMessage topButtonClick = {0};  // Close, minimize or maximize
    
    switch(message)
    {
        case WM_DESTROY:
        {
            PostQuitMessage(0);
            break;
        }
        
        // Entering modal loops will start a timer,
        // that way the system will continue to
        // dispatch messages
        case WM_ENTERSIZEMOVE:
        {
            captionClick.active = false;
            topButtonClick.active = false;
            
            // USER_TIMER_MINIMUM is 10ms, so not very precise.
            SetTimer(window, 0, USER_TIMER_MINIMUM, nullptr);
            break;
        }
        
        case WM_EXITSIZEMOVE:
        {
            KillTimer(window, 0);
            break;
        }
        
        // It's a low-priority message, so WM_SIZE
        // should be prioritized over this
        case WM_TIMER:
        {
            if(window) Win32_BreakOutOfModalLoop();
            break;
        }
        
        // Redraw immediately after resize (we don't want black bars)
        case WM_SIZE:
        {
            if(window) Win32_BreakOutOfModalLoop();
            
            // Reset the timer. Without this, the resizing is laggy
            // because we let the app update twice per resize.
            SetTimer(window, 0, USER_TIMER_MINIMUM, nullptr);
            break;
        }
        
        ///////////////////////////////////////////////////////////
        // NOTE: Clicks on the client area completely block everything;
        // not event -timer events- are sent to the window, so there's
        // just no way of executing any code while the WM_NCLBUTTONDOWN
        // message is being handled. (for example, holding down the mouse
        // button over the title bar but not moving the mouse)
        // Luckily, we can actually defer sending that message until we
        // move the mouse. As soon as we move the mouse, we'll also receive
        // WM_ENTERSIZEMOVE and the WM_TIMER messages. As for clicking on buttons,
        // we can simply not handle the WM_NCLBUTTONDOWN message at all and just
        // reimplement the functionality of the buttons on WM_NCLBUTTONUP.
        // This technique is also used in chromium and here:
        // https://github.com/glfw/glfw/pull/1426
        
        case WM_NCLBUTTONDOWN:
        {
            auto hitType = wParam;
            switch(hitType)
            {
                case HTCAPTION:
                {
                    captionClick.active = true;
                    captionClick.msg = message;
                    captionClick.lParam = lParam;
                    captionClick.wParam = wParam;
                    return 0;
                }
                case HTCLOSE:
                case HTMAXBUTTON:
                case HTMINBUTTON:
                {
                    topButtonClick.active = true;
                    topButtonClick.msg = message;
                    topButtonClick.lParam = lParam;
                    topButtonClick.wParam = wParam;
                    return 0;
                }
            }
            
            break;
        }
        
        case WM_NCMOUSEMOVE:
        case WM_MOUSEMOVE:
        {
            if(captionClick.active)
            {
                captionClick.active = false;
                
                // NOTE: There's still a bit of a stall here because this part of Windows is buggy.
                // To be more precise, when holding down the mouse button and moving the mouse very
                // slightly (a few pixels). What to do about this? Is it my responsibility to work
                // around their buggy code?
                DefWindowProc(window, captionClick.msg, captionClick.wParam, captionClick.lParam);
            }
            
            break;
        }
        
        case WM_NCLBUTTONUP:
        {
            auto hitType = wParam;
            
            if(!topButtonClick.active || topButtonClick.wParam != hitType)
                break;
            
            topButtonClick.active = false;
            
            // Reimplement the buttons' functionality, because we're
            // not handling the WM_NCLBUTTONDOWN message. This is not
            // that bad because it's only 3 buttons and we're just calling
            // the OS functions (as windows would normally do)
            switch(hitType)
            {
                case HTCLOSE:
                {
                    DestroyWindow(window);
                    break;
                }
                case HTMAXBUTTON:
                {
                    WINDOWPLACEMENT windowPlacement = {0};
                    GetWindowPlacement(window, &windowPlacement);
                    bool isMaximized = windowPlacement.showCmd == SW_SHOWMAXIMIZED;
                    
                    if(isMaximized)
                        ShowWindow(window, SW_SHOWNORMAL);
                    else
                        ShowWindow(window, SW_MAXIMIZE);
                    break;
                }
                case HTMINBUTTON:
                {
                    ShowWindow(window, SW_SHOWMINIMIZED);
                    break;
                }
            }
            
            break;
        }
        
        // Ignore right clicks (useless imo)
        case WM_NCRBUTTONDOWN:
        {
            return 0;
        }
        
        case WM_SETCURSOR:
        {
            static HCURSOR arrowCursor = LoadCursor(NULL, IDC_ARROW);
            
            // Handle cursor update when hovering over the client area
            if(LOWORD(lParam) == HTCLIENT)
            {
                SetCursor(win32.curCursor);
                return 1;
            }
            
            break;
        }
        
        // This is for user input stuff
        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
        {
            // "Capturing" means making sure
            // that the events get sent to the
            // appropriate window. Without capturing,
            // the LBUTTONUP message would not get sent,
            // thus the app wouldn't know that the user
            // released the mouse.
            if(window) SetCapture(window);
            
            Win32_BufferInputMessage(message, wParam, lParam);
            break;
        }
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
        {
            ReleaseCapture();
            
            Win32_BufferInputMessage(message, wParam, lParam);
            break;
        }
        case WM_KEYDOWN:
        case WM_KEYUP:
        {
            Win32_BufferInputMessage(message, wParam, lParam);
            break;
        }
    }
    
    return DefWindowProc(window, message, wParam, lParam);
}

void Win32_WindowEventsFiber(void* param)
{
    assert(win32.init);
    
    while(true)
    {
        win32.quit = false;
        MSG msg = {0};
        
        // Window messages
        while(PeekMessage(&msg, win32.window, 0, 0, true))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);  // This calls WndProc
        }
        
        memset(&msg, 0, sizeof(msg));
        
        // Thread messages
        while(PeekMessage(&msg, nullptr, 0, 0, true))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            
            if(msg.message == WM_QUIT) win32.quit = true;
        }
        
        // Go back to application code
        SwitchToFiber(win32.mainFiber);
    }
}

void* Win32_GetOpenGLProc(const char* name)
{
    // Load newer opengl functions via wglGetProcAddress
    void* proc = (void*)wglGetProcAddress(name);
    // These are the possible null values returned by wglGetProcAddress
    // (for some reason?)
    if(proc == 0 ||
       proc == (void*)0x1 || proc == (void*)0x2 || proc == (void*)0x3 ||
       proc == (void*)-1)
    {
        // It could be an old OpenGL 1.1 function,
        // so try to import it from the GL lib
        HMODULE module = LoadLibrary("opengl32.dll");
        proc = (void*)GetProcAddress(module, name);
    }
    
    return proc;
}

// TODO: right now this function only either returns true
// or quits the program entirely. This is reasonable for now
// but in the future we might try to use other graphics libraries
// (vulkan, direct3d) if this one can't be used for some reason.
bool Win32_SetupOpenGL()
{
    Win32_GetWGLFunctions();
    
    // Set pixel format for OpenGL context
    {
        int attrib[] =
        {
            WGL_DRAW_TO_WINDOW_ARB, GL_TRUE,
            WGL_SUPPORT_OPENGL_ARB, GL_TRUE,
            WGL_DOUBLE_BUFFER_ARB,  GL_TRUE,
            WGL_PIXEL_TYPE_ARB,     WGL_TYPE_RGBA_ARB,
            WGL_COLOR_BITS_ARB,     24,
            WGL_DEPTH_BITS_ARB,     24,
            WGL_STENCIL_BITS_ARB,   8,
            
            // Uncomment for sRGB framebuffer, from WGL_ARB_framebuffer_sRGB extension
            // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_framebuffer_sRGB.txt
            //WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB, GL_TRUE,
            
            // uncomment for multisampeld framebuffer, from WGL_ARB_multisample extension
            // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_multisample.txt
            WGL_SAMPLE_BUFFERS_ARB, 1,
            WGL_SAMPLES_ARB,        4, // 4x MSAA
            
            0
        };
        
        int format;
        UINT formats;
        int success = wgl.wglChoosePixelFormatARB(win32.windowDC, attrib, nullptr, 1, &format, &formats);
        if(!success || formats == 0)
            OS_FatalError("OpenGL does not support required pixel format.");
        
        PIXELFORMATDESCRIPTOR desc = { .nSize = sizeof(desc) };
        success = DescribePixelFormat(win32.windowDC, format, sizeof(desc), &desc);
        assert(success && "Failed to describe pixel format.");
        
        if(!SetPixelFormat(win32.windowDC, format, &desc))
            OS_FatalError("Cannot set OpenGL selected pixel format.");
    }
    
    // Create opengl context
    {
        int attrib[] =
        {
            WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
            WGL_CONTEXT_MINOR_VERSION_ARB, 6,
            WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
#ifndef NDEBUG
            WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_DEBUG_BIT_ARB,
#endif
            0
        };
        
        wgl.glContext = wgl.wglCreateContextAttribsARB(win32.windowDC, nullptr, attrib);
        if(!wgl.glContext)
            OS_FatalError("Cannot create modern OpenGL context. OpenGL 4.6 might not be supported on this architecture.");
        
        bool success = wglMakeCurrent(win32.windowDC, wgl.glContext);
        assert(success && "Failed to make current OpenGL context");
        
        wgl.wglSwapIntervalEXT(1);
    }
    
    // Load opengl functions
    if(!gladLoadGLLoader((GLADloadproc)Win32_GetOpenGLProc))
        OS_FatalError("Failed to load OpenGL.");
    
#ifndef NDEBUG
    // Enable debug callback
    glDebugMessageCallback(&OpenGLDebugCallback, NULL);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
#endif
    
    return true;
}

bool Win32_SetupD3D11()
{
    static const D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };
    
    // Create D3D11 device and context
    UINT flags = 0;
#ifndef NDEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    
    ID3D11DeviceContext* deviceContext;
    ID3D11Device* device;
    HRESULT res = D3D11CreateDevice(nullptr,
                                    D3D_DRIVER_TYPE_HARDWARE,
                                    nullptr,
                                    flags,
                                    featureLevels,
                                    ArrayCount(featureLevels),
                                    D3D11_SDK_VERSION,
                                    &device,
                                    nullptr,
                                    &deviceContext);
    
    if(FAILED(res)) return false;
    
#ifndef NDEBUG
    // For debug buils enable debug break on API errors
    {
        ID3D11InfoQueue* info;
        device->QueryInterface(IID_ID3D11InfoQueue, (void**)&info);
        info->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        info->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_ERROR, TRUE);
        info->Release();
    }
    
    // Enable debug break for DXGI
    {
        IDXGIInfoQueue* dxgiInfo;
        res = DXGIGetDebugInterface1(0, IID_IDXGIInfoQueue, (void**)&dxgiInfo);
        assert(SUCCEEDED(res));
        dxgiInfo->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        dxgiInfo->SetBreakOnSeverity(DXGI_DEBUG_ALL, DXGI_INFO_QUEUE_MESSAGE_SEVERITY_ERROR, TRUE);
        dxgiInfo->Release();
    }
#endif
    
    // Create DXGI swapchain
    IDXGISwapChain1* swapchain;
    {
        // Get DXGI device from D3D11 device
        IDXGIDevice* dxgiDevice;
        HRESULT res = device->QueryInterface(IID_IDXGIDevice, (void**)&dxgiDevice);
        assert(SUCCEEDED(res));
        
        // Get DXGI adapter from DXGI device
        IDXGIAdapter* dxgiAdapter;
        res = dxgiDevice->GetAdapter(&dxgiAdapter);
        assert(SUCCEEDED(res));
        
        // Get DXGI factory from DXGI adapter
        IDXGIFactory2* factory;
        res = dxgiAdapter->GetParent(IID_IDXGIFactory2, (void**)&factory);
        assert(SUCCEEDED(res));
        
        DXGI_SWAP_CHAIN_DESC1 swapchainDesc =
        {
            .Width = 0,
            .Height = 0,
            .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
            .SampleDesc = { 1, 0 },
            .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
            .BufferCount = 2,
            .Scaling = DXGI_SCALING_NONE,
            .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD
        };
        
        res = factory->CreateSwapChainForHwnd((IUnknown*)device, win32.window, &swapchainDesc, nullptr, nullptr, &swapchain);
        assert(SUCCEEDED(res));
        
        // Disable Alt+Enter changing monitor resolution to match window size
        factory->MakeWindowAssociation(win32.window, DXGI_MWA_NO_ALT_ENTER);
        
        factory->Release();
        dxgiAdapter->Release();
        dxgiDevice->Release();
    }
    
    d3d11.device = device;
    d3d11.deviceContext = deviceContext;
    d3d11.swapchain = swapchain;
    
    return true;
}

D3D11_Context Win32_GetD3D11Context()
{
    return d3d11;
}

// Handles the DPI stuff. This depends on the
// version of windows so it tries the newest
// API first, falling back to the older ones.
void Win32_InitDPISettings()
{
    assert(!win32.window && "Error: win32 does not allow to set process DPI settings after a window has already been created.");
    
    // TODO: Load this function dynamically
    SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
}

void OS_Init()
{
    win32.init = true;
    HINSTANCE hInst = GetModuleHandle(nullptr);
    
    // The application is aware of DPI. Otherwise windows
    // automatically scales the window up, resulting in blurry
    // visuals.
    Win32_InitDPISettings();
    
    OS_ShowCursor(true);
    
    // QPC initialization
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        win32.qpcFreq = freq.QuadPart;
    }
    
    const char* className = "Opengl Test";
    WNDCLASS wc = {0};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.style = CS_OWNDC;
    wc.lpszClassName = className;
    RegisterClass(&wc);
    
    win32.window = CreateWindowEx(0, className, "Opengl Test",                                 // Opt styles, class name and wnd name
                                  WS_OVERLAPPEDWINDOW,                                         // Window style
                                  CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,  // Size and position
                                  nullptr,                                                        // Parent window
                                  nullptr,                                                        // Menu
                                  hInst,
                                  nullptr                                                        // Additional app data
                                  );
    
    assert(win32.window && "Could not create main window.");
    
    win32.windowDC = GetDC(win32.window);
    assert(win32.windowDC && "Could not get Device Context for main window.");
    
    // TODO: update this when in fullscreen mode
    win32.usingDwm = true;
    
    // NOTE: Microsoft Insanity(tm) incoming...
    // need to create 2 separate fibers, one for
    // application code and one for window event
    // handling, to implement "proper" window
    // resizing. Yes, you've read that right.
    // See WndProc for more details.
    {
        win32.mainFiber   = ConvertThreadToFiber(0);
        win32.eventsFiber = CreateFiber(0, Win32_WindowEventsFiber, nullptr);
        
        assert(win32.mainFiber && "Could not convert main thread to fiber");
        assert(win32.eventsFiber && "Could not create window events fiber");
    }
    
    // Initialize input libraries
    {
        bool ok = Win32_LoadXInput();
    }
    
    // Initialize any graphics library.
    // On windows, it's preferred to use d3d11 over OpenGL because of buggy drivers.
    {
#ifdef GFX_OPENGL
        bool ok = Win32_SetupOpenGL();
        if(!ok) OS_FatalError("Could not set up OpenGL.");
        win32.usedGfxLib = GfxLib_OpenGL;
#elif defined(GFX_D3D11)
        bool ok = Win32_SetupD3D11();
        if(!ok) OS_FatalError("Could not set up Direct3D 11. Consider removing the 'FORCE_D3D11' flag");
        win32.usedGfxLib = GfxLib_D3D11;
#else
#error "This code path does not support this graphics library!
#endif
    }
}

void OS_ShowWindow()
{
    assert(win32.init);
    ShowWindow(win32.window, SW_NORMAL);
}

void OS_GetWindowSize(int* width, int* height)
{
    assert(win32.init);
    assert(width && height);
    
    RECT winRect = {0};
    bool ok = GetWindowRect(win32.window, &winRect);
    assert(ok);
    
    *width  = winRect.right  - winRect.left;
    *height = winRect.bottom - winRect.top;
}

void OS_GetClientAreaSize(int* width, int* height)
{
    assert(win32.init);
    assert(width && height);
    
    RECT winRect = {0};
    bool ok = GetClientRect(win32.window, &winRect);
    assert(ok);
    
    *width  = winRect.right  - winRect.left;
    *height = winRect.bottom - winRect.top;
}

void OS_SwapBuffers()
{
    assert(win32.init);
    
    // NOTE: Opengl vsync does not work properly when using
    // DWM (windowed mode)... So we have to call this
    // function which does a separate vsync. Rumors suggest
    // calling DwmFlush() immediately after SwapBuffers
    // *might* work more consistently. I love windows!
    if(win32.usedGfxLib == GfxLib_OpenGL && win32.usingDwm)
    {
        SwapBuffers(win32.windowDC);
        DwmFlush();
    }
    else
        SwapBuffers(win32.windowDC);
}

void OS_Cleanup()
{
    assert(win32.init);
    ReleaseDC(win32.window, win32.windowDC);
    DestroyWindow(win32.window);
    
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(wgl.glContext);
    
    memset(&win32, 0, sizeof(win32));
    memset(&wgl, 0, sizeof(wgl));
}

bool OS_HandleWindowEvents()
{
    assert(win32.init);
    
    // The events fiber should set quit here
    SwitchToFiber(win32.eventsFiber);
    
    return !win32.quit;
}

VirtualKeycode Win32_ConvertToCustomKeyCodes(WPARAM code)
{
    UINT aKey = 0x41;
    UINT zKey = 0x5A;
    UINT codeInt = (UINT)code;
    if(codeInt >= aKey && codeInt <= zKey)
        return (VirtualKeycode)(codeInt - aKey + Keycode_A);
    
    VirtualKeycode res = Keycode_Null;
    switch(code)
    {
        case VK_LBUTTON: res = Keycode_LMouse; break;
        case VK_RBUTTON: res = Keycode_RMouse; break;
        case VK_MBUTTON: res = Keycode_MMouse; break;
        default:         res = Keycode_Null;   break;
    }
    
    return res;
}

void Win32_HandleWindowMessageRange(UINT min, UINT max)
{
    MSG msg = {0};
    while(PeekMessage(&msg, win32.window, min, max, true))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

OS_InputState OS_PollInput()
{
    assert(win32.init);
    static_assert(XUSER_MAX_COUNT <= MaxActiveControllers);
    
    OS_InputState res = {0};
    
    // Gamepad controllers first
    for(int i = 0; i < XUSER_MAX_COUNT; ++i)
    {
        XINPUT_STATE state = {0};
        DWORD active = XInputGetState(i, &state);
        if(active == ERROR_SUCCESS)
        {
            XINPUT_GAMEPAD* pad = &state.Gamepad;
            OS_GamepadState& resPad = res.gamepads[i];
            resPad.active = true;
            
            // Mapping between XInput buttons and our buttons
            if(pad->wButtons & XINPUT_GAMEPAD_DPAD_UP)        resPad.buttons |= Gamepad_Dpad_Up;
            if(pad->wButtons & XINPUT_GAMEPAD_DPAD_DOWN)      resPad.buttons |= Gamepad_Dpad_Down;
            if(pad->wButtons & XINPUT_GAMEPAD_DPAD_LEFT)      resPad.buttons |= Gamepad_Dpad_Left;
            if(pad->wButtons & XINPUT_GAMEPAD_DPAD_RIGHT)     resPad.buttons |= Gamepad_Dpad_Right;
            if(pad->wButtons & XINPUT_GAMEPAD_START)          resPad.buttons |= Gamepad_Start;
            if(pad->wButtons & XINPUT_GAMEPAD_BACK)           resPad.buttons |= Gamepad_Back;
            if(pad->wButtons & XINPUT_GAMEPAD_LEFT_THUMB)     resPad.buttons |= Gamepad_LeftThumb;
            if(pad->wButtons & XINPUT_GAMEPAD_RIGHT_THUMB)    resPad.buttons |= Gamepad_RightThumb;
            if(pad->wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER)  resPad.buttons |= Gamepad_LeftShoulder;
            if(pad->wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER) resPad.buttons |= Gamepad_RightShoulder;
            if(pad->wButtons & XINPUT_GAMEPAD_A)              resPad.buttons |= Gamepad_A;
            if(pad->wButtons & XINPUT_GAMEPAD_B)              resPad.buttons |= Gamepad_B;
            if(pad->wButtons & XINPUT_GAMEPAD_X)              resPad.buttons |= Gamepad_X;
            if(pad->wButtons & XINPUT_GAMEPAD_Y)              resPad.buttons |= Gamepad_Y;
            
            resPad.leftTrigger  = (float)pad->bLeftTrigger  / 255;
            resPad.rightTrigger = (float)pad->bRightTrigger / 255;
            
            // For the left and right stick, normalize values differently
            // if it's positive/negative because the range is different
            resPad.leftStickX  = (float)pad->sThumbLX / (pad->sThumbLX > 0 ? 32767 : 32768);
            resPad.leftStickY  = (float)pad->sThumbLY / (pad->sThumbLY > 0 ? 32767 : 32768);
            resPad.rightStickX = (float)pad->sThumbRX / (pad->sThumbRX > 0 ? 32767 : 32768);
            resPad.rightStickY = (float)pad->sThumbRY / (pad->sThumbRY > 0 ? 32767 : 32768);
        }
    }
    
    // Mouse state
    {
        POINT p;
        GetCursorPos(&p);  // In screen coordinates
        if(!ScreenToClient(win32.window, &p))
        {
            res.mouse.active = false;
            res.mouse.xPos   = -1;
            res.mouse.yPos   = -1;
        }
        else
        {
            res.mouse.active = p.x >= 0 && p.y >= 0;
            res.mouse.xPos   = p.x;
            res.mouse.yPos   = p.y;
        }
        
        // Buttons and clicks
        
        // In a modal loop we won't get click events anyway
        // so...
        if(!win32.isInModalLoop)
        {
            // Get all input messages to update the input buffers
            Win32_HandleWindowMessageRange(WM_KEYDOWN, WM_KEYUP);
            Win32_HandleWindowMessageRange(WM_LBUTTONDOWN, WM_LBUTTONUP);
            Win32_HandleWindowMessageRange(WM_RBUTTONDOWN, WM_RBUTTONUP);
            Win32_HandleWindowMessageRange(WM_MBUTTONDOWN, WM_MBUTTONUP);
        }
        
        // This holds state for holding down buttons
        static bool heldKeys[Keycode_Count] = {0};
        bool pressedKeys[Keycode_Count] = {0};
        
        for(int i = 0; i < win32.numInputs; ++i)
        {
            Win32_Message msg = win32.bufferedInputs[i];
            switch(msg.kind)
            {
                case WM_KEYDOWN:
                {
                    VirtualKeycode keycode = Win32_ConvertToCustomKeyCodes(msg.wParam);
                    heldKeys[keycode] = true;
                    pressedKeys[keycode] = true;
                    break;
                }
                case WM_KEYUP:
                {
                    VirtualKeycode keycode = Win32_ConvertToCustomKeyCodes(msg.wParam);
                    heldKeys[keycode] = false;
                    break;
                }
                
                // For some reason WM_KEYDOWN doesn't work
                // with mouse clicks even though there are virtual
                // keycodes for just that. Not documented.
                case WM_LBUTTONDOWN:
                {
                    heldKeys[Keycode_LMouse] = true;
                    pressedKeys[Keycode_LMouse] = true;
                    break;
                }
                case WM_LBUTTONUP:
                {
                    heldKeys[Keycode_LMouse] = false;
                    break;
                }
                case WM_RBUTTONDOWN:
                {
                    heldKeys[Keycode_RMouse] = true;
                    pressedKeys[Keycode_RMouse] = true;
                    break;
                }
                case WM_RBUTTONUP:
                {
                    heldKeys[Keycode_RMouse] = false;
                    break;
                }
                case WM_MBUTTONDOWN:
                {
                    heldKeys[Keycode_MMouse] = true;
                    pressedKeys[Keycode_MMouse] = true;
                    break;
                }
                case WM_MBUTTONUP:
                {
                    heldKeys[Keycode_MMouse] = false;
                    break;
                }
            }
        }
        
        // Clear buffered inputs
        memset(&win32.bufferedInputs, 0, sizeof(win32.bufferedInputs));
        win32.numInputs = 0;
        
        // held, pressed: true
        // not held, pressed: true
        // held, not pressed: true
        // not held, not pressed: false
        for(int i = 0; i < Keycode_Count; ++i)
            res.virtualKeys[i] = heldKeys[i] || pressedKeys[i];
    }
    
    return res;
}

void OS_ShowCursor(bool show)
{
    assert(win32.init);
    
    static HCURSOR arrowCursor = LoadCursor(nullptr, IDC_ARROW);
    win32.curCursor = show ? arrowCursor : nullptr;
    
    bool isInvisible = GetCursor() == nullptr;
    // Only set the cursor if we're trying to modify visibility
    if((show && isInvisible) || (!show && !isInvisible))
        SetCursor(win32.curCursor);
}

void OS_SetCursorPos(int64_t posX, int64_t posY)
{
    POINT p = {0};
    p.x = posX;
    p.y = posY;
    ClientToScreen(win32.window, &p); // To screen coordinates
    SetCursorPos(p.x, p.y);
}

void OS_FatalError(const char* message)
{
    MessageBox(nullptr, message, "Error", MB_ICONEXCLAMATION);
    ExitProcess(0);
}

uint64_t OS_GetTicks()
{
    LARGE_INTEGER ticks;
    QueryPerformanceCounter(&ticks);
    return ticks.QuadPart;
}

double OS_GetElapsedSeconds(uint64_t startTicks, uint64_t endTicks)
{
    assert(win32.init);
    
    // Get number of microseconds elapsed
    LARGE_INTEGER elapsedMicros;
    elapsedMicros.QuadPart = endTicks - startTicks;
    return elapsedMicros.QuadPart / (double)win32.qpcFreq;
}