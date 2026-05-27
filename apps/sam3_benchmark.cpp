#include "cli_common.hpp"

int main(int argc, char** argv) {
  return RunManifestCli(argc, argv, "models/manifests/sam3_full.yaml", "sam3_benchmark");
}
