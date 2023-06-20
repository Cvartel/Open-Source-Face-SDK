#ifndef MESHFITTER_H
#define MESHFITTER_H

#include <tdv/modules/ONNXModule.h>
#include <tdv/modules/BaseEstimationModule.h>

namespace tdv {

namespace modules {

class MeshFitterModule : public BaseEstimationModule
{
public:
	MeshFitterModule(const tdv::data::Context& config);
private:
	friend class ONNXModule<MeshFitterModule>;
};

}
}


#endif // MESHFITTER_H
