// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEPoolingLayer.h>
#include "arm_converter/arm_converter.hpp"


namespace ArmPlugin {

template<typename Pool>
static void FillLayerInfo(const Pool& node, arm_compute::PoolingLayerInfo& pool_info) {
    unsigned int pad_left   = node.get_pads_begin().at(D2::W);
    unsigned int pad_right  = node.get_pads_end().at(D2::W);
    unsigned int pad_top    = node.get_pads_begin().at(D2::H);
    unsigned int pad_bottom = node.get_pads_end().at(D2::H);
    unsigned int kernel_w   = node.get_kernel().at(D2::W);
    unsigned int kernel_h   = node.get_kernel().at(D2::H);
    unsigned int stride_x   = node.get_strides().at(D2::W);
    unsigned int stride_y   = node.get_strides().at(D2::H);

    arm_compute::DimensionRoundingType round = (node.get_rounding_type() == ngraph::op::RoundingType::FLOOR)
                                             ? arm_compute::DimensionRoundingType::FLOOR
                                             : arm_compute::DimensionRoundingType::CEIL;

    pool_info.data_layout       = arm_compute::DataLayout::NCHW;
    pool_info.pool_size         = arm_compute::Size2D(kernel_w, kernel_h);
    pool_info.pad_stride_info   = arm_compute::PadStrideInfo(stride_x, stride_y, pad_left, pad_right, pad_top, pad_bottom, round);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::MaxPool& node) {
    arm_compute::PoolingLayerInfo pool_info;
    FillLayerInfo(node, pool_info);
    pool_info.pool_type = arm_compute::PoolingType::MAX;
    return MakeConversion<arm_compute::NEPoolingLayer>(node.input(0), node.output(0), pool_info);
}

template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v8::MaxPool& node) {
    auto make = [&] (auto refFunction) {
    return this->MakeConversion(refFunction,
                                node.input(0),
                                node.output(0),
                                node.output(1)
                                node.get_input_shape(0),
                                node.get_output_shape(0),
                                node.get_kernel_shape(),
                                node.get_kernel(),
                                node.get_strides(),
                                node.get_dilations(),
                                node.get_pads_begin(),
                                node.get_pads_end());
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::max_pool),
        node.input(0), allTypes,
        node.output(1), indexTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::AvgPool& node) {
    arm_compute::PoolingLayerInfo pool_info;
    FillLayerInfo(node, pool_info);
    pool_info.pool_type       = arm_compute::PoolingType::AVG;
    pool_info.exclude_padding = node.get_exclude_pad();
    return MakeConversion<arm_compute::NEPoolingLayer>(node.input(0), node.output(0), pool_info);
}
}  // namespace ArmPlugin
