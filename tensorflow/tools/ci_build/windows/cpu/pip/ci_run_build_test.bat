rem This script helps set up the necessary environment variables to run TF-Windows build & unit tests
rem The actual folder/file paths can be locally changed by user to suit one's need.
rem The current values are meant to work with Jenkins/CI setup for GCP-Windows machines
rem We also assume a few Software components are also installed in the machine: MS VC++, MINGW SYS64, Python 3.8, JAVA, Git, Bazelisk etc.

rem Assuming, the user has already git cloned the TF repo to %WORKSPACE%\tensorflow (aka %TF_LOCATION%)
rem Assuming, the user has already created a Python 3.8 virtual env at location %WSVENVDIR%
rem We also assume, a set of Python modules (pip) are installed, e.g.
rem absl-py astunparse flatbuffers google_pasta h5py keras-nightly keras_preprocessing numpy opt_einsum protobuf scipy six termcolor typing_extensions wheel wrapt gast tensorboard tf-estimator-nightly packaging portpicker

rem first run this separately in cmd prompt:
rem C:\Jenkins\workspace\tensorflow-eigen-test-win\venv38\Scripts\activate
rem Then run this bat file to set env vars => C:\...\...\setup_venv38.bat
rem Then invoke build/test scripts (see comments at end of this script)

set WORKSPACE=C:\Jenkins\workspace\tensorflow-eigen-test-win
set PYTHON_VERSION=38
set WSVENVDIR=%WORKSPACE%\venv%PYTHON_VERSION%
for /f "delims=" %%i in ('cygpath -m %WSVENVDIR%') do set WSVENVDIRUNIX=%%i
set TMP=C:\tmp
set TEMP=C:\tmp
rem TMPDIR is needed for TF's tmp work
set TMPDIR=C:\tmp\tftmp
set BAZEL_LOCATION=C:\Tools
set USE_BAZEL_VERSION=5.1.0
set "TF_VC_VERSION=16.6"
set VS_LOCATION=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
set JAVA_LOCATION=C:\Program Files\Eclipse Adoptium\jdk-11.0.14.101-hotspot


set PATH=C:\Tools;%BAZEL_LOCATION%;%VENV_LOC%\Scripts;C:\Program Files\Eclipse Adoptium\jdk-11.0.14.101-hotspot\bin;C:\Program Files\Python38\Scripts\;C:\Program Files\Python38\;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\System32\OpenSSH\;C:\ProgramData\GooGet;C:\Program Files\Google\Compute Engine\metadata_scripts;C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin;C:\Program Files\PowerShell\7\;C:\Program Files\Google\Compute Engine\sysprep;C:\msys64;C:\Program Files\Git\cmd;C:\msys64\usr\bin;C:\Program Files\dotnet\;C:\Program Files\Microsoft SQL Server\130\Tools\Binn\;C:\Program Files\Microsoft SQL Server\Client SDK\ODBC\170\Tools\Binn\;C:\Program Files\Bazel-5.0.0;C:\Users\mlp_admin\AppData\Local\Microsoft\WindowsApps;C:\Users\mlp_admin\.dotnet\tools;

set PYTHONPATH=%WSVENVDIR%\Lib\site-packages
for /f "delims=" %%i in ('cygpath -m %PYTHONPATH%') do set PYTHONPATH=%%i
set PYTHON_BIN_PATH=%WSVENVDIRUNIX%/Scripts/python.exe
set PYTHON_LIB_PATH=%WSVENVDIRUNIX%/Lib/site-packages
set PYTHON_DIRECTORY=%WSVENVDIRUNIX%
rem strip c:/ from beginning
set PYTHON_DIRECTORY=%PYTHON_DIRECTORY:~3%
set PYTHON_EXE=%WSVENVDIRUNIX%/Scripts/python.exe

echo "PYTHON_VERSION: %PYTHON_VERSION%"
echo "BAZEL_LOCATION: %BAZEL_LOCATION%"
set "PYTHON_LOCATION=%PYTHON_DIRECTORY%"
set "TF_LOCATION=%WORKSPACE%\tensorflow"
set "BAZEL_SH=C:\msys64\usr\bin\bash.exe"
set "BAZEL_VS=%VS_LOCATION%"
set "BAZEL_VC=%VS_LOCATION%\VC"

set "PYTHON_DIRECTORY=%PYTHON_LOCATION%"
set "MKL_DIR=%WORKSPACE%\mkl_output"

set "JAVA_HOME=%JAVA_LOCATION%"
set "SYS_PATH=%SYSTEM_PATH%"
echo "=========================================="
echo "Environment variables:"
echo "run_mkl         : %run_mkl%"
echo "TF_LOCATION     : %TF_LOCATION%"
echo "PYTHON_DIRECTORY: %PYTHON_DIRECTORY%"
echo "BAZEL_SH        : %BAZEL_SH%"
echo "BAZEL_VS        : %BAZEL_VS%"
echo "BAZEL_VC        : %BAZEL_VC%"
echo "MKL_DIR         : %MKL_DIR%"
echo "TF_VC_VERSION   : %TF_VC_VERSION%"
echo "JAVA_HOME       : %JAVA_LOCATION%"
echo "PATH            : %PATH%"
echo "=========================================="

where python
python --version
where bazel
cd %TF_LOCATION%
bazel version

rem echo "Updating PYTHON_BIN_PATH & PYTHON_LIB_PATH env vars in common_env.sh"
rem echo export PYTHON_BIN_PATH=%PYTHON_BIN_PATH% >> %WORKSPACE%\tensorflow\tensorflow\tools\ci_build\windows\bazel\common_env.sh
rem echo export PYTHON_LIB_PATH=%PYTHON_LIB_PATH% >> %WORKSPACE%\tensorflow\tensorflow\tools\ci_build\windows\bazel\common_env.sh

rem yes "" | python configure.py
rem C:\Jenkins\workspace\tensorflow-eigen-test-win\tensorflow> bash -l c:\Jenkins\workspace\tensorflow-eigen-test-win\tensorflow/tensorflow/tools/ci_build/windows/cpu/pip/build_tf_windows.sh --extra_build_flags "--action_env=TEMP=C:\tmp --action_env=TMP=C:\tmp" --extra_test_flags "--action_env=TEMP=C:\tmp --action_env=TMP=C:\tmp"  >C:\tmp\build.log 2>&1

yes "" | python configure.py
bash -l %WORKSPACE%\tensorflow/tensorflow/tools/ci_build/windows/cpu/pip/build_tf_windows.sh --extra_build_flags "--action_env=TEMP=%TMP% --action_env=TMP=%TMP%" --extra_test_flags "--action_env=TEMP=%TMP% --action_env=TMP=%TMP%"
