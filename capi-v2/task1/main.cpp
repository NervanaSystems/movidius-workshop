#include <mvnc.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {

	struct ncDeviceHandle_t* deviceHandle = NULL;
	std::cout << "Creating NCS resources" << std::endl;
	int index = 0;  // Index of device to query for
  ncStatus_t ret = ncDeviceCreate(index,&deviceHandle); // hardcoded max name size 
  if (ret == NC_OK) {
    std::cout << "Found NCS named: " << index++ << " OK" << std::endl;
  }

	// If not devices present the exit
	if (index == 0) {
		std::cerr << std::string("Error: No Intel Movidius identified in a system!\n");
		exit(-1);
	}

	// Using first devicoftma
	ret = ncDeviceOpen(deviceHandle);
	if(ret == NC_OK) {

		ret = ncDeviceClose(deviceHandle);
		if (ret != NC_OK) {
			std::cerr << "Error: Closing of device: "<<  std::to_string(index-1) <<"failed!" << std::endl;
		}
	} else {
		// If we cannot open communication with device then clean up resources
		std::cerr << std::string("Error: Could not open NCS device: ") + std::to_string(index-1) ;
	}
	

	std::cout << "Destroying NCS resources" << std::endl;
	ret = ncDeviceDestroy(&deviceHandle);
	if (ret != NC_OK) {
		std::cerr << "Error: Freeing resources of device: "<<  std::to_string(index-1) <<"failed!" << std::endl;
	}

  return 0;
}
