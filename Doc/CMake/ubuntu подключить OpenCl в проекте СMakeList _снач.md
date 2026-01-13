<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# ubuntu подключить OpenCl в проекте СMakeList

сначало найти есть ли на компьютере

Отлично. Теперь создам полный гайд с примерами.

## Проверка и подключение OpenCL в Ubuntu

### Шаг 1: Проверьте, установлена ли OpenCL

Сначала проверьте наличие OpenCL на вашей машине через терминал:

```bash
clinfo
```

Если команда не найдена или не выводит информацию о платформах/устройствах, OpenCL не установлена или её рантайм отсутствует.

### Шаг 2: Установите OpenCL заголовки и библиотеки

Самый простой способ — через APT:[^1]

```bash
sudo apt update
sudo apt install opencl-headers ocl-icd-opencl-dev -y
```

Это установит:

- **opencl-headers** — C/C++ заголовки из Khronos
- **ocl-icd-opencl-dev** — ICD Loader (installable Client Driver Loader) с libOpenCL.so[^1]

Проверьте установку:

```bash
clinfo
```

Если вывод похож на:

```
Number of platforms: 1
Platform 0: ...
  Device 0: ...
```

OpenCL готова к использованию.

### Шаг 3: CMakeLists.txt с OpenCL

Вот минимальный пример для C++ проекта:[^1]

```cmake
cmake_minimum_required(VERSION 3.15)

project(MyOpenCLApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ищет OpenCL
find_package(OpenCL REQUIRED)

add_executable(MyOpenCLApp
    src/main.cpp
)

# Линкует OpenCL библиотеку
target_link_libraries(MyOpenCLApp PRIVATE OpenCL::OpenCL)

# Опционально: дефайн версии OpenCL API
target_compile_definitions(MyOpenCLApp PRIVATE CL_TARGET_OPENCL_VERSION=120)
```

**Ключевые моменты:**

- `find_package(OpenCL REQUIRED)` — CMake сам ищет OpenCL в стандартных местах (`/usr/include`, `/usr/lib`).[^1]
- `OpenCL::OpenCL` — modern target-based линковка (CMake 3.15+).[^1]
- `CL_TARGET_OPENCL_VERSION` — какую версию API использовать (100 = 1.0, 120 = 1.2 и т.д.).[^1]


### Шаг 4: Сконфигурируйте и соберите

В папке с `CMakeLists.txt`:

```bash
cmake -S . -B ./build
cmake --build ./build --config Release
```

Если нет ошибок — готово. Запустите:

```bash
./build/MyOpenCLApp
```


### Специальные случаи

**Если OpenCL не нашлась в стандартных местах:**

Укажите путь явно:

```cmake
find_package(OpenCL REQUIRED HINTS /usr/lib /usr/include)
```

**Если у вас NVIDIA GPU (CUDA):**

CMake 3.17+ может найти OpenCL из CUDA:

```cmake
cmake_minimum_required(VERSION 3.17)
project(MyOpenCLApp LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
add_executable(MyOpenCLApp src/main.cpp)
target_link_libraries(MyOpenCLApp PRIVATE CUDA::OpenCL)
```

**Если у вас AMD GPU:**

Установите AMD ROCm:

```bash
sudo apt install rocm-opencl rocm-opencl-dev -y
```

Затем используйте обычный `find_package(OpenCL REQUIRED)`.

### Простой пример C++ кода для проверки

```cpp
#include <CL/cl.hpp>
#include <iostream>

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    std::cout << "Found " << platforms.size() << " OpenCL platform(s)" << std::endl;
    
    for (const auto& platform : platforms) {
        std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        std::cout << "  Devices: " << devices.size() << std::endl;
    }
    
    return 0;
}
```

Скомпилируйте с `cmake` как выше и запустите — вы увидите все доступные OpenCL платформы и устройства.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/getting_started_linux.md

[^2]: https://streamhpc.com/blog/2011-06-24/install-opencl-on-debianubuntu-orderly/

[^3]: https://stackoverflow.com/questions/67641589/how-can-i-tell-cmake-3-9-i-want-to-use-nvidias-opencl-library

[^4]: https://stackoverflow.com/questions/76318281/how-to-get-cmake-to-find-opencl-if-it-was-installed-via-vcpkg

[^5]: https://chromium.googlesource.com/external/github.com/KhronosGroup/OpenCL-CTS/+/12fce84a66602c33cf909b4a47911d7085962661/CMakeLists.txt

[^6]: https://devtalk.blender.org/t/guide-install-amd-opencl-on-pop-os-some-ubuntu-derivates-linux-amdgpu-amdgpu-pro-rocm-rocr/13458

[^7]: https://forums.developer.nvidia.com/t/cmake-findopencl-cmake-module-to-build-opencl-apps/9695

[^8]: https://e2e.ti.com/support/processors-group/processors/f/processors-forum/542984/how-to-compile-opencl-examples-source-with-cmake

[^9]: https://stackoverflow.com/questions/63942556/opencl-development-under-ubuntu

[^10]: https://community.intel.com/t5/OpenCL-for-CPU/intel-opencl-SDK-How-do-I-know-I-installed-successfully/td-p/1085340

[^11]: https://github.com/mantiuk/opencl_examples/blob/master/CMakeLists.txt

[^12]: https://forums.developer.nvidia.com/t/ubuntu-16-10-cuda-toolkit-install-with-opencl/46070

[^13]: https://support.zivid.com/getting-started/software-installation/gpu/install-opencl-drivers-ubuntu.html

[^14]: https://gist.github.com/jarutis/a64eaa38c1caaf7bc3d28cea64bb8359

[^15]: https://community.intel.com/t5/OpenCL-for-CPU/How-to-install-OpenCL-Runtime-for-Ubuntu-18-04/m-p/1468856

