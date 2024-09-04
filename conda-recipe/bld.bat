mkdir build
cd build

REM ----------------------------------------------------------------------
IF NOT DEFINED WITH_CPLEX (SET WITH_CPLEX=0)
IF NOT DEFINED WITH_GUROBI (SET WITH_GUROBI=0)
IF NOT DEFINED CPLEX_ROOT_DIR (SET CPLEX_ROOT_DIR="")
IF NOT DEFINED GUROBI_ROOT_DIR (SET GUROBI_ROOT_DIR="")
IF %WITH_CPLEX% == "" (SET WITH_CPLEX=0)
IF %WITH_GUROBI% == "" (SET WITH_GUROBI=0)

SET OPTIMIZER_ARGS=
IF "%WITH_GUROBI%" == "1" (
    IF NOT DEFINED GUROBI_ROOT_DIR (
        ECHO "GUROBI_ROOT_DIR must be set for building!"
        exit 1
    )
    IF "%GUROBI_ROOT_DIR%" == "" (
        ECHO "GUROBI_ROOT_DIR cannot be empty for building!"
        exit 1
    ) ELSE (
        ECHO "Using GUROBI_ROOT_DIR=%GUROBI_ROOT_DIR%"
    )
    IF NOT DEFINED GUROBI_ROOT_DIR (
        ECHO "GUROBI_LIB_WIN must be set for building!"
        exit 1
    )
    IF "%GUROBI_LIB_WIN%" == "" (
        ECHO "GUROBI_LIB_WIN cannot be empty for building!"
        exit 1
    )
    REM if we build with Gurobi, we need to configure the paths.
    REM The GUROBI_ROOT_DIR should point to gurobiXYZ\win64
    ECHO "found gurobi lib %GUROBI_LIB_WIN%"
    :: ensure single double quotes with :"=
    SET OPTIMIZER_ARGS=-DWITH_GUROBI=ON -DGUROBI_ROOT_DIR="%GUROBI_ROOT_DIR:"=%" ^
      -DGUROBI_LIBRARY="%GUROBI_LIB_WIN:"=%" -DGUROBI_INCLUDE_DIR="%GUROBI_ROOT_DIR:"=%\include" ^
      -DGUROBI_CPP_LIBRARY="%GUROBI_ROOT_DIR:"=%\lib\gurobi_c++md2015.lib"
    rem set GUROBI_LIB=%%i
    rem dir "%GUROBI_ROOT_DIR%\lib\gurobi*.lib" /s/b|findstr gurobi[0-9][0-9].lib>gurobilib.tmp
    rem set /p GUROBI_LIB=<gurobilib.tmp
)

IF "%WITH_CPLEX%" == "1" (
    :: ensure single double quotes with :"=
    SET OPTIMIZER_ARGS=-DWITH_CPLEX=ON -DCPLEX_ROOT_DIR="%CPLEX_ROOT_DIR:"=%" -DCPLEX_WIN_VERSION=%CPLEX_WIN_VERSION%
)

REM ----------------------------------------------------------------------

set CONFIGURATION=Release

cmake .. -G "%CMAKE_GENERATOR%" -DCMAKE_PREFIX_PATH="%LIBRARY_PREFIX%" ^
    -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%"  ^
    -DBOOST_ROOT="%LIBRARY%" ^
    -DCMAKE_CXX_FLAGS="-DBOOST_ALL_NO_LIB /EHsc" ^
    ^
    %OPTIMIZER_ARGS% ^
    ^
    -DPython_EXECUTABLE=%PYTHON% ^
    -DBUILD_NIFTY_PYTHON=yes ^
    -DBUILD_CPP_TEST=no ^
    -DBUILD_CPP_EXAMPLES=no ^
    -DWITH_GLPK=no ^
    -DWITH_HDF5=yes ^
    -DWITH_LP_MP=no ^
    -DWITH_QPBO=no

cmake --build . --target ALL_BUILD --config %CONFIGURATION%
if errorlevel 1 exit 1
cmake --build . --target INSTALL --config %CONFIGURATION%
if errorlevel 1 exit 1


IF "%WITH_GUROBI%" == "1" (
    REM Rename the nifty package to 'nifty_with_gurobi'
    cd "%SP_DIR%"
    rename nifty nifty_with_gurobi
)

IF "%WITH_CPLEX%" == "1" (
    REM Rename the nifty package to 'nifty_with_cplex'
    cd "%SP_DIR%"
    rename nifty nifty_with_cplex
)
