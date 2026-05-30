#include "sam_s600/cli/manifest_cli.hpp"

int main(int argc, char** argv) {
  return RunManifestCli(argc, argv, "configs/manifests/sam3_image.yaml", "sam3_image");
}
