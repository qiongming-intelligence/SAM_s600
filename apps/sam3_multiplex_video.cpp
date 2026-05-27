#include "cli_common.hpp"

int main(int argc, char** argv) {
  return RunManifestCli(argc, argv, "models/manifests/sam3_multiplex.yaml", "sam3_multiplex_video");
}
