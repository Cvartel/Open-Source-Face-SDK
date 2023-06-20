#include <tdv/modules/MeshFitterModule.h>
#include <tdv/inference/MeshFitterInference.h>


namespace tdv {

namespace modules {


MeshFitterModule::MeshFitterModule(const tdv::data::Context& config):
	BaseEstimationModule(config)
{
	block = std::unique_ptr<ProcessingBlock>(new tdv::modules::MeshFitterInference<MeshFitterModule>(config));
}

}
}
