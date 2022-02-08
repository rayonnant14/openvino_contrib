// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <ngraph/runtime/reference/shape_of.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v3::ShapeOf& node) {
    std::cout << "node " << std::endl;
    // return;
    // auto make = [&] (auto refFunction) {
    // return this->MakeConversion(refFunction,
    //                             node.input(0),
    //                             node.output(0));
    // };
    // return CallSwitch(
    //     AP_WRAP(make, ngraph::runtime::reference::shape_of),
    //     node.get_output_element_type(0), indexTypes);
}

} // namespace ArmPlugin