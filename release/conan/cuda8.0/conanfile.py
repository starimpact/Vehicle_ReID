from conans import ConanFile, CMake

class Mxnet(ConanFile): 
    name = "dgVehicleReID"          #name 
    version = "0.2.0"    #version 
    settings = "os", "compiler", "build_type", "arch" 
    exports = "*"
    
    def package(self):
        self.copy("*", dst="include", src="include")
        self.copy("*", dst="lib", src="lib")

    def package_info(self):
        self.cpp_info.libs = ["dgVehicleReID"]


