>       ??@?	   j???Abrain.Event:2R$
"tensorflow.core.util.events_writerݐ??L?     V?o:	??5j???A"??
j
input_1Placeholder*
dtype0*
shape:?	*'
_output_shapes
:?	
?
max_pooling2d/MaxPoolMaxPoolinput_1*
explicit_paddings
 *
T0*
ksize
*'
_output_shapes
:?*
strides
*
data_formatNHWC*
paddingVALID
?
8stream/kernel/Initializer/stateless_random_uniform/shapeConst*
dtype0* 
_class
loc:@stream/kernel*%
valueB"            *
_output_shapes
:
?
6stream/kernel/Initializer/stateless_random_uniform/minConst* 
_class
loc:@stream/kernel*
dtype0*
valueB
 *   ?*
_output_shapes
: 
?
6stream/kernel/Initializer/stateless_random_uniform/maxConst*
dtype0* 
_class
loc:@stream/kernel*
valueB
 *   ?*
_output_shapes
: 
?
Tstream/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst* 
_class
loc:@stream/kernel*
dtype0*
valueB"??0    *
_output_shapes
:
?
Ostream/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterTstream/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed* 
_class
loc:@stream/kernel*
Tseed0* 
_output_shapes
::
?
Ostream/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst* 
_class
loc:@stream/kernel*
dtype0*
value	B :*
_output_shapes
: 
?
Kstream/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV28stream/kernel/Initializer/stateless_random_uniform/shapeOstream/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterQstream/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1Ostream/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
Tshape0* 
_class
loc:@stream/kernel*&
_output_shapes
:
?
6stream/kernel/Initializer/stateless_random_uniform/subSub6stream/kernel/Initializer/stateless_random_uniform/max6stream/kernel/Initializer/stateless_random_uniform/min* 
_class
loc:@stream/kernel*
T0*
_output_shapes
: 
?
6stream/kernel/Initializer/stateless_random_uniform/mulMulKstream/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV26stream/kernel/Initializer/stateless_random_uniform/sub* 
_class
loc:@stream/kernel*
T0*&
_output_shapes
:
?
2stream/kernel/Initializer/stateless_random_uniformAddV26stream/kernel/Initializer/stateless_random_uniform/mul6stream/kernel/Initializer/stateless_random_uniform/min* 
_class
loc:@stream/kernel*
T0*&
_output_shapes
:
?
stream/kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: * 
_class
loc:@stream/kernel*
shape:*
shared_namestream/kernel
k
.stream/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream/kernel*
_output_shapes
: 
?
stream/kernel/AssignAssignVariableOpstream/kernel2stream/kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
w
!stream/kernel/Read/ReadVariableOpReadVariableOpstream/kernel*
dtype0*&
_output_shapes
:
?
stream/bias/Initializer/zerosConst*
dtype0*
_class
loc:@stream/bias*
valueB*    *
_output_shapes
:
?
stream/biasVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *
_class
loc:@stream/bias*
shape:*
shared_namestream/bias
g
,stream/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream/bias*
_output_shapes
: 
u
stream/bias/AssignAssignVariableOpstream/biasstream/bias/Initializer/zeros*
dtype0*
validate_shape( 
g
stream/bias/Read/ReadVariableOpReadVariableOpstream/bias*
dtype0*
_output_shapes
:
y
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOpstream/kernel*
dtype0*&
_output_shapes
:
?
stream/conv2d/Conv2DConv2Dmax_pooling2d/MaxPool#stream/conv2d/Conv2D/ReadVariableOp*
	dilations
*
explicit_paddings
 *
T0*
use_cudnn_on_gpu(*&
_output_shapes
:{*
strides
*
data_formatNHWC*
paddingVALID
l
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOpstream/bias*
dtype0*
_output_shapes
:
?
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D$stream/conv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:{
?
*batch_normalization/gamma/Initializer/onesConst*
dtype0*,
_class"
 loc:@batch_normalization/gamma*
valueB*  ??*
_output_shapes
:
?
batch_normalization/gammaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *,
_class"
 loc:@batch_normalization/gamma*
shape:**
shared_namebatch_normalization/gamma
?
:batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/gamma*
_output_shapes
: 
?
 batch_normalization/gamma/AssignAssignVariableOpbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
dtype0*
_output_shapes
:
?
*batch_normalization/beta/Initializer/zerosConst*
dtype0*+
_class!
loc:@batch_normalization/beta*
valueB*    *
_output_shapes
:
?
batch_normalization/betaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *+
_class!
loc:@batch_normalization/beta*
shape:*)
shared_namebatch_normalization/beta
?
9batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
?
batch_normalization/beta/AssignAssignVariableOpbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
dtype0*
_output_shapes
:
?
1batch_normalization/moving_mean/Initializer/zerosConst*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
valueB*    *
_output_shapes
:
?
batch_normalization/moving_meanVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *2
_class(
&$loc:@batch_normalization/moving_mean*
shape:*0
shared_name!batch_normalization/moving_mean
?
@batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
?
&batch_normalization/moving_mean/AssignAssignVariableOpbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
:
?
4batch_normalization/moving_variance/Initializer/onesConst*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
valueB*  ??*
_output_shapes
:
?
#batch_normalization/moving_varianceVarHandleOp*
	container *
allowed_devices
 *6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
?
Dbatch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
?
*batch_normalization/moving_variance/AssignAssignVariableOp#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
:
\
keras_learning_phase/inputConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
shape: *
_output_shapes
: 
?
batch_normalization/condIfkeras_learning_phasebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancestream/conv2d/BiasAdd*&
_read_only_resource_inputs
*4
else_branch%R#
!batch_normalization_cond_false_44*3
then_branch$R"
 batch_normalization_cond_true_43*@
_output_shapes.
,:{::: : : : : : : *
Tcond0
*
_lower_using_switch_merge(*?
output_shapes.
,:{::: : : : : : : *
Tout
2
*
Tin	
2
x
!batch_normalization/cond/IdentityIdentitybatch_normalization/cond*
T0*&
_output_shapes
:{
p
#batch_normalization/cond/Identity_1Identitybatch_normalization/cond:1*
T0*
_output_shapes
:
p
#batch_normalization/cond/Identity_2Identitybatch_normalization/cond:2*
T0*
_output_shapes
:
l
#batch_normalization/cond/Identity_3Identitybatch_normalization/cond:3*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_4Identitybatch_normalization/cond:4*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_5Identitybatch_normalization/cond:5*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_6Identitybatch_normalization/cond:6*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_7Identitybatch_normalization/cond:7*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_8Identitybatch_normalization/cond:8*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_9Identitybatch_normalization/cond:9*
T0*
_output_shapes
: 
?
batch_normalization/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *
output_shapes
: *6
then_branch'R%
#batch_normalization_cond_1_true_111*
_output_shapes
: *
Tcond0
*
_lower_using_switch_merge(*7
else_branch(R&
$batch_normalization_cond_1_false_112*
Tout
2*	
Tin
 
l
#batch_normalization/cond_1/IdentityIdentitybatch_normalization/cond_1*
T0*
_output_shapes
: 
?
)batch_normalization/AssignMovingAvg/sub/xConst*
dtype0*2
_class(
&$loc:@batch_normalization/moving_mean*
valueB
 *  ??*
_output_shapes
: 
?
'batch_normalization/AssignMovingAvg/subSub)batch_normalization/AssignMovingAvg/sub/x#batch_normalization/cond_1/Identity*2
_class(
&$loc:@batch_normalization/moving_mean*
T0*
_output_shapes
: 
?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
:
?
)batch_normalization/AssignMovingAvg/sub_1Sub2batch_normalization/AssignMovingAvg/ReadVariableOp#batch_normalization/cond/Identity_1*2
_class(
&$loc:@batch_normalization/moving_mean*
T0*
_output_shapes
:
?
'batch_normalization/AssignMovingAvg/mulMul)batch_normalization/AssignMovingAvg/sub_1'batch_normalization/AssignMovingAvg/sub*2
_class(
&$loc:@batch_normalization/moving_mean*
T0*
_output_shapes
:
?
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/mul*2
_class(
&$loc:@batch_normalization/moving_mean*&
 _has_manual_control_dependencies(*
dtype0
?
4batch_normalization/AssignMovingAvg/ReadVariableOp_1ReadVariableOpbatch_normalization/moving_mean8^batch_normalization/AssignMovingAvg/AssignSubVariableOp*
dtype0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
?
+batch_normalization/AssignMovingAvg_1/sub/xConst*
dtype0*6
_class,
*(loc:@batch_normalization/moving_variance*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization/AssignMovingAvg_1/subSub+batch_normalization/AssignMovingAvg_1/sub/x#batch_normalization/cond_1/Identity*6
_class,
*(loc:@batch_normalization/moving_variance*
T0*
_output_shapes
: 
?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
:
?
+batch_normalization/AssignMovingAvg_1/sub_1Sub4batch_normalization/AssignMovingAvg_1/ReadVariableOp#batch_normalization/cond/Identity_2*6
_class,
*(loc:@batch_normalization/moving_variance*
T0*
_output_shapes
:
?
)batch_normalization/AssignMovingAvg_1/mulMul+batch_normalization/AssignMovingAvg_1/sub_1)batch_normalization/AssignMovingAvg_1/sub*6
_class,
*(loc:@batch_normalization/moving_variance*
T0*
_output_shapes
:
?
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/mul*6
_class,
*(loc:@batch_normalization/moving_variance*&
 _has_manual_control_dependencies(*
dtype0
?
6batch_normalization/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp#batch_normalization/moving_variance:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
_output_shapes
:
k
activation/ReluRelu!batch_normalization/cond/Identity*
T0*&
_output_shapes
:{
?
Dstream_1/depthwise_kernel/Initializer/stateless_random_uniform/shapeConst*
dtype0*,
_class"
 loc:@stream_1/depthwise_kernel*%
valueB"            *
_output_shapes
:
?
Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/minConst*
dtype0*,
_class"
 loc:@stream_1/depthwise_kernel*
valueB
 *?7?*
_output_shapes
: 
?
Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/maxConst*,
_class"
 loc:@stream_1/depthwise_kernel*
dtype0*
valueB
 *?7?*
_output_shapes
: 
?
`stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*,
_class"
 loc:@stream_1/depthwise_kernel*
dtype0*
valueB"P?\    *
_output_shapes
:
?
[stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter`stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*,
_class"
 loc:@stream_1/depthwise_kernel*
Tseed0* 
_output_shapes
::
?
[stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*,
_class"
 loc:@stream_1/depthwise_kernel*
dtype0*
value	B :*
_output_shapes
: 
?
Wstream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Dstream_1/depthwise_kernel/Initializer/stateless_random_uniform/shape[stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter]stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1[stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
Tshape0*,
_class"
 loc:@stream_1/depthwise_kernel*&
_output_shapes
:
?
Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/subSubBstream_1/depthwise_kernel/Initializer/stateless_random_uniform/maxBstream_1/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_1/depthwise_kernel*
T0*
_output_shapes
: 
?
Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/mulMulWstream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/sub*,
_class"
 loc:@stream_1/depthwise_kernel*
T0*&
_output_shapes
:
?
>stream_1/depthwise_kernel/Initializer/stateless_random_uniformAddV2Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/mulBstream_1/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_1/depthwise_kernel*
T0*&
_output_shapes
:
?
stream_1/depthwise_kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *,
_class"
 loc:@stream_1/depthwise_kernel*
shape:**
shared_namestream_1/depthwise_kernel
?
:stream_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_1/depthwise_kernel*
_output_shapes
: 
?
 stream_1/depthwise_kernel/AssignAssignVariableOpstream_1/depthwise_kernel>stream_1/depthwise_kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
?
-stream_1/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_1/depthwise_kernel*
dtype0*&
_output_shapes
:
?
stream_1/bias/Initializer/zerosConst* 
_class
loc:@stream_1/bias*
dtype0*
valueB*    *
_output_shapes
:
?
stream_1/biasVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: * 
_class
loc:@stream_1/bias*
shape:*
shared_namestream_1/bias
k
.stream_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_1/bias*
_output_shapes
: 
{
stream_1/bias/AssignAssignVariableOpstream_1/biasstream_1/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_1/bias/Read/ReadVariableOpReadVariableOpstream_1/bias*
dtype0*
_output_shapes
:
?
2stream_1/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpstream_1/depthwise_kernel*
dtype0*&
_output_shapes
:
?
)stream_1/depthwise_conv2d/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
1stream_1/depthwise_conv2d/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
#stream_1/depthwise_conv2d/depthwiseDepthwiseConv2dNativeactivation/Relu2stream_1/depthwise_conv2d/depthwise/ReadVariableOp*
	dilations
*
explicit_paddings
 *
T0*&
_output_shapes
:<*
strides
*
data_formatNHWC*
paddingVALID
z
0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpstream_1/bias*
dtype0*
_output_shapes
:
?
!stream_1/depthwise_conv2d/BiasAddBiasAdd#stream_1/depthwise_conv2d/depthwise0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:<
?
,batch_normalization_1/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
valueB*  ??*
_output_shapes
:
?
batch_normalization_1/gammaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_1/gamma*
shape:*,
shared_namebatch_normalization_1/gamma
?
<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
?
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_1/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_1/beta*
dtype0*
valueB*    *
_output_shapes
:
?
batch_normalization_1/betaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_1/beta*
shape:*+
shared_namebatch_normalization_1/beta
?
;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
?
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_1/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
valueB*    *
_output_shapes
:
?
!batch_normalization_1/moving_meanVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_1/moving_mean*
shape:*2
shared_name#!batch_normalization_1/moving_mean
?
Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
?
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_1/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
valueB*  ??*
_output_shapes
:
?
%batch_normalization_1/moving_varianceVarHandleOp*
	container *
allowed_devices
 *8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
?
Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
?
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_1/condIfkeras_learning_phasebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!stream_1/depthwise_conv2d/BiasAdd*&
_read_only_resource_inputs
*?
output_shapes.
,:<::: : : : : : : *6
then_branch'R%
#batch_normalization_1_cond_true_172*@
_output_shapes.
,:<::: : : : : : : *
Tcond0
*
_lower_using_switch_merge(*7
else_branch(R&
$batch_normalization_1_cond_false_173*
Tout
2
*
Tin	
2
|
#batch_normalization_1/cond/IdentityIdentitybatch_normalization_1/cond*
T0*&
_output_shapes
:<
t
%batch_normalization_1/cond/Identity_1Identitybatch_normalization_1/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_1/cond/Identity_2Identitybatch_normalization_1/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_1/cond/Identity_3Identitybatch_normalization_1/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_4Identitybatch_normalization_1/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_5Identitybatch_normalization_1/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_6Identitybatch_normalization_1/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_7Identitybatch_normalization_1/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_8Identitybatch_normalization_1/cond:8*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_9Identitybatch_normalization_1/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_1/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *9
else_branch*R(
&batch_normalization_1_cond_1_false_241*8
then_branch)R'
%batch_normalization_1_cond_1_true_240*
_output_shapes
: *
Tcond0
*
_lower_using_switch_merge(*
output_shapes
: *
Tout
2*	
Tin
 
p
%batch_normalization_1/cond_1/IdentityIdentitybatch_normalization_1/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_1/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization_1/AssignMovingAvg/subSub+batch_normalization_1/AssignMovingAvg/sub/x%batch_normalization_1/cond_1/Identity*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_1/AssignMovingAvg/sub_1Sub4batch_normalization_1/AssignMovingAvg/ReadVariableOp%batch_normalization_1/cond/Identity_1*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0*
_output_shapes
:
?
)batch_normalization_1/AssignMovingAvg/mulMul+batch_normalization_1/AssignMovingAvg/sub_1)batch_normalization_1/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0*
_output_shapes
:
?
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_1/moving_mean*&
 _has_manual_control_dependencies(*
dtype0
?
6batch_normalization_1/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
?
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*
dtype0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB
 *  ??*
_output_shapes
: 
?
+batch_normalization_1/AssignMovingAvg_1/subSub-batch_normalization_1/AssignMovingAvg_1/sub/x%batch_normalization_1/cond_1/Identity*8
_class.
,*loc:@batch_normalization_1/moving_variance*
T0*
_output_shapes
: 
?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%batch_normalization_1/cond/Identity_2*8
_class.
,*loc:@batch_normalization_1/moving_variance*
T0*
_output_shapes
:
?
+batch_normalization_1/AssignMovingAvg_1/mulMul-batch_normalization_1/AssignMovingAvg_1/sub_1+batch_normalization_1/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_1/moving_variance*
T0*
_output_shapes
:
?
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*
dtype0*&
 _has_manual_control_dependencies(*8
_class.
,*loc:@batch_normalization_1/moving_variance
?
8batch_normalization_1/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:
o
activation_1/ReluRelu#batch_normalization_1/cond/Identity*
T0*&
_output_shapes
:<
?
Dstream_2/depthwise_kernel/Initializer/stateless_random_uniform/shapeConst*,
_class"
 loc:@stream_2/depthwise_kernel*
dtype0*%
valueB"            *
_output_shapes
:
?
Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/minConst*,
_class"
 loc:@stream_2/depthwise_kernel*
dtype0*
valueB
 *?7?*
_output_shapes
: 
?
Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/maxConst*,
_class"
 loc:@stream_2/depthwise_kernel*
dtype0*
valueB
 *?7?*
_output_shapes
: 
?
`stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
dtype0*,
_class"
 loc:@stream_2/depthwise_kernel*
valueB"?.o    *
_output_shapes
:
?
[stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter`stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*,
_class"
 loc:@stream_2/depthwise_kernel*
Tseed0* 
_output_shapes
::
?
[stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*
dtype0*,
_class"
 loc:@stream_2/depthwise_kernel*
value	B :*
_output_shapes
: 
?
Wstream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Dstream_2/depthwise_kernel/Initializer/stateless_random_uniform/shape[stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter]stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1[stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
Tshape0*,
_class"
 loc:@stream_2/depthwise_kernel*&
_output_shapes
:
?
Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/subSubBstream_2/depthwise_kernel/Initializer/stateless_random_uniform/maxBstream_2/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_2/depthwise_kernel*
T0*
_output_shapes
: 
?
Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/mulMulWstream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/sub*,
_class"
 loc:@stream_2/depthwise_kernel*
T0*&
_output_shapes
:
?
>stream_2/depthwise_kernel/Initializer/stateless_random_uniformAddV2Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/mulBstream_2/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_2/depthwise_kernel*
T0*&
_output_shapes
:
?
stream_2/depthwise_kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *,
_class"
 loc:@stream_2/depthwise_kernel*
shape:**
shared_namestream_2/depthwise_kernel
?
:stream_2/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_2/depthwise_kernel*
_output_shapes
: 
?
 stream_2/depthwise_kernel/AssignAssignVariableOpstream_2/depthwise_kernel>stream_2/depthwise_kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
?
-stream_2/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_2/depthwise_kernel*
dtype0*&
_output_shapes
:
?
stream_2/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@stream_2/bias*
valueB*    *
_output_shapes
:
?
stream_2/biasVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: * 
_class
loc:@stream_2/bias*
shape:*
shared_namestream_2/bias
k
.stream_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_2/bias*
_output_shapes
: 
{
stream_2/bias/AssignAssignVariableOpstream_2/biasstream_2/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_2/bias/Read/ReadVariableOpReadVariableOpstream_2/bias*
dtype0*
_output_shapes
:
?
4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpstream_2/depthwise_kernel*
dtype0*&
_output_shapes
:
?
+stream_2/depthwise_conv2d_1/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
3stream_2/depthwise_conv2d_1/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
%stream_2/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativeactivation_1/Relu4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp*
	dilations
*
explicit_paddings
 *
T0*&
_output_shapes
:9*
strides
*
data_formatNHWC*
paddingVALID
|
2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpstream_2/bias*
dtype0*
_output_shapes
:
?
#stream_2/depthwise_conv2d_1/BiasAddBiasAdd%stream_2/depthwise_conv2d_1/depthwise2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:9
?
,batch_normalization_2/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
valueB*  ??*
_output_shapes
:
?
batch_normalization_2/gammaVarHandleOp*
	container *
allowed_devices
 *.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_2/gamma
?
<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
?
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_2/beta/Initializer/zerosConst*
dtype0*-
_class#
!loc:@batch_normalization_2/beta*
valueB*    *
_output_shapes
:
?
batch_normalization_2/betaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_2/beta*
shape:*+
shared_namebatch_normalization_2/beta
?
;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
?
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_2/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
valueB*    *
_output_shapes
:
?
!batch_normalization_2/moving_meanVarHandleOp*
	container *
allowed_devices
 *4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean
?
Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
?
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_2/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
valueB*  ??*
_output_shapes
:
?
%batch_normalization_2/moving_varianceVarHandleOp*
	container *
allowed_devices
 *8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance
?
Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
?
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_2/condIfkeras_learning_phasebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#stream_2/depthwise_conv2d_1/BiasAdd*&
_read_only_resource_inputs
*?
output_shapes.
,:9::: : : : : : : *6
then_branch'R%
#batch_normalization_2_cond_true_301*@
_output_shapes.
,:9::: : : : : : : *
Tcond0
*
_lower_using_switch_merge(*7
else_branch(R&
$batch_normalization_2_cond_false_302*
Tout
2
*
Tin	
2
|
#batch_normalization_2/cond/IdentityIdentitybatch_normalization_2/cond*
T0*&
_output_shapes
:9
t
%batch_normalization_2/cond/Identity_1Identitybatch_normalization_2/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_2/cond/Identity_2Identitybatch_normalization_2/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_2/cond/Identity_3Identitybatch_normalization_2/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_4Identitybatch_normalization_2/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_5Identitybatch_normalization_2/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_6Identitybatch_normalization_2/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_7Identitybatch_normalization_2/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_8Identitybatch_normalization_2/cond:8*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_9Identitybatch_normalization_2/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_2/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *9
else_branch*R(
&batch_normalization_2_cond_1_false_370*8
then_branch)R'
%batch_normalization_2_cond_1_true_369*
_output_shapes
: *
Tcond0
*
_lower_using_switch_merge(*
output_shapes
: *
Tout
2*	
Tin
 
p
%batch_normalization_2/cond_1/IdentityIdentitybatch_normalization_2/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_2/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization_2/AssignMovingAvg/subSub+batch_normalization_2/AssignMovingAvg/sub/x%batch_normalization_2/cond_1/Identity*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_2/AssignMovingAvg/sub_1Sub4batch_normalization_2/AssignMovingAvg/ReadVariableOp%batch_normalization_2/cond/Identity_1*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0*
_output_shapes
:
?
)batch_normalization_2/AssignMovingAvg/mulMul+batch_normalization_2/AssignMovingAvg/sub_1)batch_normalization_2/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0*
_output_shapes
:
?
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*
dtype0*&
 _has_manual_control_dependencies(*4
_class*
(&loc:@batch_normalization_2/moving_mean
?
6batch_normalization_2/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_2/moving_mean:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
?
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
+batch_normalization_2/AssignMovingAvg_1/subSub-batch_normalization_2/AssignMovingAvg_1/sub/x%batch_normalization_2/cond_1/Identity*8
_class.
,*loc:@batch_normalization_2/moving_variance*
T0*
_output_shapes
: 
?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%batch_normalization_2/cond/Identity_2*8
_class.
,*loc:@batch_normalization_2/moving_variance*
T0*
_output_shapes
:
?
+batch_normalization_2/AssignMovingAvg_1/mulMul-batch_normalization_2/AssignMovingAvg_1/sub_1+batch_normalization_2/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_2/moving_variance*
T0*
_output_shapes
:
?
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*
dtype0*&
 _has_manual_control_dependencies(*8
_class.
,*loc:@batch_normalization_2/moving_variance
?
8batch_normalization_2/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_2/moving_variance<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
o
activation_2/ReluRelu#batch_normalization_2/cond/Identity*
T0*&
_output_shapes
:9
?
"stream_3/average_pooling2d/AvgPoolAvgPoolactivation_2/Relu*
strides
*
ksize
*
T0*
data_formatNHWC*
paddingVALID*&
_output_shapes
:
?
Dstream_4/depthwise_kernel/Initializer/stateless_random_uniform/shapeConst*
dtype0*,
_class"
 loc:@stream_4/depthwise_kernel*%
valueB"            *
_output_shapes
:
?
Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/minConst*,
_class"
 loc:@stream_4/depthwise_kernel*
dtype0*
valueB
 *׳ݾ*
_output_shapes
: 
?
Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/maxConst*,
_class"
 loc:@stream_4/depthwise_kernel*
dtype0*
valueB
 *׳?>*
_output_shapes
: 
?
`stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
dtype0*,
_class"
 loc:@stream_4/depthwise_kernel*
valueB"t'I    *
_output_shapes
:
?
[stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter`stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*,
_class"
 loc:@stream_4/depthwise_kernel*
Tseed0* 
_output_shapes
::
?
[stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*,
_class"
 loc:@stream_4/depthwise_kernel*
dtype0*
value	B :*
_output_shapes
: 
?
Wstream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Dstream_4/depthwise_kernel/Initializer/stateless_random_uniform/shape[stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter]stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1[stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*,
_class"
 loc:@stream_4/depthwise_kernel*
Tshape0*
dtype0*&
_output_shapes
:
?
Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/subSubBstream_4/depthwise_kernel/Initializer/stateless_random_uniform/maxBstream_4/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_4/depthwise_kernel*
T0*
_output_shapes
: 
?
Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/mulMulWstream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/sub*,
_class"
 loc:@stream_4/depthwise_kernel*
T0*&
_output_shapes
:
?
>stream_4/depthwise_kernel/Initializer/stateless_random_uniformAddV2Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/mulBstream_4/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_4/depthwise_kernel*
T0*&
_output_shapes
:
?
stream_4/depthwise_kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *,
_class"
 loc:@stream_4/depthwise_kernel*
shape:**
shared_namestream_4/depthwise_kernel
?
:stream_4/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_4/depthwise_kernel*
_output_shapes
: 
?
 stream_4/depthwise_kernel/AssignAssignVariableOpstream_4/depthwise_kernel>stream_4/depthwise_kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
?
-stream_4/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_4/depthwise_kernel*
dtype0*&
_output_shapes
:
?
stream_4/bias/Initializer/zerosConst* 
_class
loc:@stream_4/bias*
dtype0*
valueB*    *
_output_shapes
:
?
stream_4/biasVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: * 
_class
loc:@stream_4/bias*
shape:*
shared_namestream_4/bias
k
.stream_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_4/bias*
_output_shapes
: 
{
stream_4/bias/AssignAssignVariableOpstream_4/biasstream_4/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_4/bias/Read/ReadVariableOpReadVariableOpstream_4/bias*
dtype0*
_output_shapes
:
?
4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpstream_4/depthwise_kernel*
dtype0*&
_output_shapes
:
?
+stream_4/depthwise_conv2d_2/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
3stream_4/depthwise_conv2d_2/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
%stream_4/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative"stream_3/average_pooling2d/AvgPool4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp*
	dilations
*
explicit_paddings
 *
T0*&
_output_shapes
:*
strides
*
data_formatNHWC*
paddingVALID
|
2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpstream_4/bias*
dtype0*
_output_shapes
:
?
#stream_4/depthwise_conv2d_2/BiasAddBiasAdd%stream_4/depthwise_conv2d_2/depthwise2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:
?
,batch_normalization_3/gamma/Initializer/onesConst*
dtype0*.
_class$
" loc:@batch_normalization_3/gamma*
valueB*  ??*
_output_shapes
:
?
batch_normalization_3/gammaVarHandleOp*
	container *
allowed_devices
 *.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma
?
<batch_normalization_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
?
"batch_normalization_3/gamma/AssignAssignVariableOpbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_3/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_3/beta*
dtype0*
valueB*    *
_output_shapes
:
?
batch_normalization_3/betaVarHandleOp*
	container *
allowed_devices
 *-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta
?
;batch_normalization_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
?
!batch_normalization_3/beta/AssignAssignVariableOpbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_3/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
valueB*    *
_output_shapes
:
?
!batch_normalization_3/moving_meanVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_3/moving_mean*
shape:*2
shared_name#!batch_normalization_3/moving_mean
?
Bbatch_normalization_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
?
(batch_normalization_3/moving_mean/AssignAssignVariableOp!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_3/moving_variance/Initializer/onesConst*
dtype0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB*  ??*
_output_shapes
:
?
%batch_normalization_3/moving_varianceVarHandleOp*
	container *
allowed_devices
 *8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance
?
Fbatch_normalization_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
?
,batch_normalization_3/moving_variance/AssignAssignVariableOp%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_3/condIfkeras_learning_phasebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#stream_4/depthwise_conv2d_2/BiasAdd*&
_read_only_resource_inputs
*7
else_branch(R&
$batch_normalization_3_cond_false_432*6
then_branch'R%
#batch_normalization_3_cond_true_431*@
_output_shapes.
,:::: : : : : : : *
Tcond0
*
_lower_using_switch_merge(*?
output_shapes.
,:::: : : : : : : *
Tout
2
*
Tin	
2
|
#batch_normalization_3/cond/IdentityIdentitybatch_normalization_3/cond*
T0*&
_output_shapes
:
t
%batch_normalization_3/cond/Identity_1Identitybatch_normalization_3/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_3/cond/Identity_2Identitybatch_normalization_3/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_3/cond/Identity_3Identitybatch_normalization_3/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_4Identitybatch_normalization_3/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_5Identitybatch_normalization_3/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_6Identitybatch_normalization_3/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_7Identitybatch_normalization_3/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_8Identitybatch_normalization_3/cond:8*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_9Identitybatch_normalization_3/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_3/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *9
else_branch*R(
&batch_normalization_3_cond_1_false_500*8
then_branch)R'
%batch_normalization_3_cond_1_true_499*
_output_shapes
: *
Tcond0
*
_lower_using_switch_merge(*
output_shapes
: *
Tout
2*	
Tin
 
p
%batch_normalization_3/cond_1/IdentityIdentitybatch_normalization_3/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_3/AssignMovingAvg/sub/xConst*
dtype0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization_3/AssignMovingAvg/subSub+batch_normalization_3/AssignMovingAvg/sub/x%batch_normalization_3/cond_1/Identity*4
_class*
(&loc:@batch_normalization_3/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_3/AssignMovingAvg/sub_1Sub4batch_normalization_3/AssignMovingAvg/ReadVariableOp%batch_normalization_3/cond/Identity_1*4
_class*
(&loc:@batch_normalization_3/moving_mean*
T0*
_output_shapes
:
?
)batch_normalization_3/AssignMovingAvg/mulMul+batch_normalization_3/AssignMovingAvg/sub_1)batch_normalization_3/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_3/moving_mean*
T0*
_output_shapes
:
?
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*
dtype0*&
 _has_manual_control_dependencies(*4
_class*
(&loc:@batch_normalization_3/moving_mean
?
6batch_normalization_3/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
?
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*
dtype0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB
 *  ??*
_output_shapes
: 
?
+batch_normalization_3/AssignMovingAvg_1/subSub-batch_normalization_3/AssignMovingAvg_1/sub/x%batch_normalization_3/cond_1/Identity*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0*
_output_shapes
: 
?
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp%batch_normalization_3/cond/Identity_2*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0*
_output_shapes
:
?
+batch_normalization_3/AssignMovingAvg_1/mulMul-batch_normalization_3/AssignMovingAvg_1/sub_1+batch_normalization_3/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0*
_output_shapes
:
?
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*
dtype0*&
 _has_manual_control_dependencies(*8
_class.
,*loc:@batch_normalization_3/moving_variance
?
8batch_normalization_3/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_3/moving_variance<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
o
activation_3/ReluRelu#batch_normalization_3/cond/Identity*
T0*&
_output_shapes
:
?
Dstream_5/depthwise_kernel/Initializer/stateless_random_uniform/shapeConst*,
_class"
 loc:@stream_5/depthwise_kernel*
dtype0*%
valueB"            *
_output_shapes
:
?
Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/minConst*
dtype0*,
_class"
 loc:@stream_5/depthwise_kernel*
valueB
 *b???*
_output_shapes
: 
?
Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/maxConst*
dtype0*,
_class"
 loc:@stream_5/depthwise_kernel*
valueB
 *b??>*
_output_shapes
: 
?
`stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
dtype0*,
_class"
 loc:@stream_5/depthwise_kernel*
valueB"k?    *
_output_shapes
:
?
[stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter`stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*,
_class"
 loc:@stream_5/depthwise_kernel*
Tseed0* 
_output_shapes
::
?
[stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*,
_class"
 loc:@stream_5/depthwise_kernel*
dtype0*
value	B :*
_output_shapes
: 
?
Wstream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Dstream_5/depthwise_kernel/Initializer/stateless_random_uniform/shape[stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter]stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1[stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
Tshape0*,
_class"
 loc:@stream_5/depthwise_kernel*&
_output_shapes
:
?
Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/subSubBstream_5/depthwise_kernel/Initializer/stateless_random_uniform/maxBstream_5/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_5/depthwise_kernel*
T0*
_output_shapes
: 
?
Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/mulMulWstream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/sub*,
_class"
 loc:@stream_5/depthwise_kernel*
T0*&
_output_shapes
:
?
>stream_5/depthwise_kernel/Initializer/stateless_random_uniformAddV2Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/mulBstream_5/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_5/depthwise_kernel*
T0*&
_output_shapes
:
?
stream_5/depthwise_kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *,
_class"
 loc:@stream_5/depthwise_kernel*
shape:**
shared_namestream_5/depthwise_kernel
?
:stream_5/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_5/depthwise_kernel*
_output_shapes
: 
?
 stream_5/depthwise_kernel/AssignAssignVariableOpstream_5/depthwise_kernel>stream_5/depthwise_kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
?
-stream_5/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_5/depthwise_kernel*
dtype0*&
_output_shapes
:
?
stream_5/bias/Initializer/zerosConst* 
_class
loc:@stream_5/bias*
dtype0*
valueB*    *
_output_shapes
:
?
stream_5/biasVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: * 
_class
loc:@stream_5/bias*
shape:*
shared_namestream_5/bias
k
.stream_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_5/bias*
_output_shapes
: 
{
stream_5/bias/AssignAssignVariableOpstream_5/biasstream_5/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_5/bias/Read/ReadVariableOpReadVariableOpstream_5/bias*
dtype0*
_output_shapes
:
?
4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpstream_5/depthwise_kernel*
dtype0*&
_output_shapes
:
?
+stream_5/depthwise_conv2d_3/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
3stream_5/depthwise_conv2d_3/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
%stream_5/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativeactivation_3/Relu4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp*
	dilations
*
explicit_paddings
 *
T0*&
_output_shapes
:*
strides
*
data_formatNHWC*
paddingVALID
|
2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpstream_5/bias*
dtype0*
_output_shapes
:
?
#stream_5/depthwise_conv2d_3/BiasAddBiasAdd%stream_5/depthwise_conv2d_3/depthwise2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:
?
,batch_normalization_4/gamma/Initializer/onesConst*
dtype0*.
_class$
" loc:@batch_normalization_4/gamma*
valueB*  ??*
_output_shapes
:
?
batch_normalization_4/gammaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_4/gamma*
shape:*,
shared_namebatch_normalization_4/gamma
?
<batch_normalization_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
?
"batch_normalization_4/gamma/AssignAssignVariableOpbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_4/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
valueB*    *
_output_shapes
:
?
batch_normalization_4/betaVarHandleOp*
	container *
allowed_devices
 *-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta
?
;batch_normalization_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
?
!batch_normalization_4/beta/AssignAssignVariableOpbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_4/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
valueB*    *
_output_shapes
:
?
!batch_normalization_4/moving_meanVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_4/moving_mean*
shape:*2
shared_name#!batch_normalization_4/moving_mean
?
Bbatch_normalization_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
?
(batch_normalization_4/moving_mean/AssignAssignVariableOp!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_4/moving_variance/Initializer/onesConst*
dtype0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
valueB*  ??*
_output_shapes
:
?
%batch_normalization_4/moving_varianceVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_4/moving_variance*
shape:*6
shared_name'%batch_normalization_4/moving_variance
?
Fbatch_normalization_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
?
,batch_normalization_4/moving_variance/AssignAssignVariableOp%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_4/condIfkeras_learning_phasebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#stream_5/depthwise_conv2d_3/BiasAdd*&
_read_only_resource_inputs
*?
output_shapes.
,:::: : : : : : : *6
then_branch'R%
#batch_normalization_4_cond_true_560*@
_output_shapes.
,:::: : : : : : : *
Tcond0
*
_lower_using_switch_merge(*7
else_branch(R&
$batch_normalization_4_cond_false_561*
Tout
2
*
Tin	
2
|
#batch_normalization_4/cond/IdentityIdentitybatch_normalization_4/cond*
T0*&
_output_shapes
:
t
%batch_normalization_4/cond/Identity_1Identitybatch_normalization_4/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_4/cond/Identity_2Identitybatch_normalization_4/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_4/cond/Identity_3Identitybatch_normalization_4/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_4Identitybatch_normalization_4/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_5Identitybatch_normalization_4/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_6Identitybatch_normalization_4/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_7Identitybatch_normalization_4/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_8Identitybatch_normalization_4/cond:8*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_9Identitybatch_normalization_4/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_4/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *9
else_branch*R(
&batch_normalization_4_cond_1_false_629*8
then_branch)R'
%batch_normalization_4_cond_1_true_628*
_output_shapes
: *
Tcond0
*
_lower_using_switch_merge(*
output_shapes
: *
Tout
2*	
Tin
 
p
%batch_normalization_4/cond_1/IdentityIdentitybatch_normalization_4/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_4/AssignMovingAvg/sub/xConst*
dtype0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization_4/AssignMovingAvg/subSub+batch_normalization_4/AssignMovingAvg/sub/x%batch_normalization_4/cond_1/Identity*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_4/AssignMovingAvg/sub_1Sub4batch_normalization_4/AssignMovingAvg/ReadVariableOp%batch_normalization_4/cond/Identity_1*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0*
_output_shapes
:
?
)batch_normalization_4/AssignMovingAvg/mulMul+batch_normalization_4/AssignMovingAvg/sub_1)batch_normalization_4/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0*
_output_shapes
:
?
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_4/moving_mean*&
 _has_manual_control_dependencies(*
dtype0
?
6batch_normalization_4/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_4/moving_mean:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp*
dtype0*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:
?
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*
dtype0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
valueB
 *  ??*
_output_shapes
: 
?
+batch_normalization_4/AssignMovingAvg_1/subSub-batch_normalization_4/AssignMovingAvg_1/sub/x%batch_normalization_4/cond_1/Identity*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0*
_output_shapes
: 
?
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp%batch_normalization_4/cond/Identity_2*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0*
_output_shapes
:
?
+batch_normalization_4/AssignMovingAvg_1/mulMul-batch_normalization_4/AssignMovingAvg_1/sub_1+batch_normalization_4/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0*
_output_shapes
:
?
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*
dtype0*&
 _has_manual_control_dependencies(*8
_class.
,*loc:@batch_normalization_4/moving_variance
?
8batch_normalization_4/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_4/moving_variance<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
o
activation_4/ReluRelu#batch_normalization_4/cond/Identity*
T0*&
_output_shapes
:
?
dropout/condIfkeras_learning_phaseactivation_4/Relu* 
_read_only_resource_inputs
 *3
output_shapes"
 :: : : : : : : *(
then_branchR
dropout_cond_true_649*4
_output_shapes"
 :: : : : : : : *
Tcond0
*
_lower_using_switch_merge(*)
else_branchR
dropout_cond_false_650*
Tout

2*
Tin
2
`
dropout/cond/IdentityIdentitydropout/cond*
T0*&
_output_shapes
:
T
dropout/cond/Identity_1Identitydropout/cond:1*
T0*
_output_shapes
: 
T
dropout/cond/Identity_2Identitydropout/cond:2*
T0*
_output_shapes
: 
T
dropout/cond/Identity_3Identitydropout/cond:3*
T0*
_output_shapes
: 
T
dropout/cond/Identity_4Identitydropout/cond:4*
T0*
_output_shapes
: 
T
dropout/cond/Identity_5Identitydropout/cond:5*
T0*
_output_shapes
: 
T
dropout/cond/Identity_6Identitydropout/cond:6*
T0*
_output_shapes
: 
T
dropout/cond/Identity_7Identitydropout/cond:7*
T0*
_output_shapes
: 
g
stream_6/flatten/ConstConst*
dtype0*
valueB"????0   *
_output_shapes
:
?
stream_6/flatten/ReshapeReshapedropout/cond/Identitystream_6/flatten/Const*
Tshape0*
T0*
_output_shapes

:0
?
7dense/kernel/Initializer/stateless_random_uniform/shapeConst*
_class
loc:@dense/kernel*
dtype0*
valueB"0   
   *
_output_shapes
:
?
5dense/kernel/Initializer/stateless_random_uniform/minConst*
_class
loc:@dense/kernel*
dtype0*
valueB
 *.???*
_output_shapes
: 
?
5dense/kernel/Initializer/stateless_random_uniform/maxConst*
dtype0*
_class
loc:@dense/kernel*
valueB
 *.??>*
_output_shapes
: 
?
Sdense/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_class
loc:@dense/kernel*
dtype0*
valueB"?ß
    *
_output_shapes
:
?
Ndense/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterSdense/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*
_class
loc:@dense/kernel*
Tseed0* 
_output_shapes
::
?
Ndense/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*
dtype0*
_class
loc:@dense/kernel*
value	B :*
_output_shapes
: 
?
Jdense/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV27dense/kernel/Initializer/stateless_random_uniform/shapeNdense/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterPdense/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1Ndense/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
Tshape0*
_class
loc:@dense/kernel*
_output_shapes

:0

?
5dense/kernel/Initializer/stateless_random_uniform/subSub5dense/kernel/Initializer/stateless_random_uniform/max5dense/kernel/Initializer/stateless_random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
: 
?
5dense/kernel/Initializer/stateless_random_uniform/mulMulJdense/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV25dense/kernel/Initializer/stateless_random_uniform/sub*
_class
loc:@dense/kernel*
T0*
_output_shapes

:0

?
1dense/kernel/Initializer/stateless_random_uniformAddV25dense/kernel/Initializer/stateless_random_uniform/mul5dense/kernel/Initializer/stateless_random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes

:0

?
dense/kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel*
shape
:0
*
shared_namedense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
?
dense/kernel/AssignAssignVariableOpdense/kernel1dense/kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:0

?
dense/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense/bias*
valueB
*    *
_output_shapes
:

?

dense/biasVarHandleOp*
	container *
allowed_devices
 *
_class
loc:@dense/bias*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
r
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0*
validate_shape( 
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:

h
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:0

?
dense/MatMulMatMulstream_6/flatten/Reshapedense/MatMul/ReadVariableOp*
transpose_b( *
T0*
transpose_a( *
_output_shapes

:

c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:

?
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*
_output_shapes

:

J

dense/ReluReludense/BiasAdd*
T0*
_output_shapes

:

?
9dense_1/kernel/Initializer/stateless_random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
dtype0*
valueB"
      *
_output_shapes
:
?
7dense_1/kernel/Initializer/stateless_random_uniform/minConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB
 *?5?*
_output_shapes
: 
?
7dense_1/kernel/Initializer/stateless_random_uniform/maxConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB
 *?5?*
_output_shapes
: 
?
Udense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*!
_class
loc:@dense_1/kernel*
dtype0*
valueB"???0    *
_output_shapes
:
?
Pdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterUdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*!
_class
loc:@dense_1/kernel*
Tseed0* 
_output_shapes
::
?
Pdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*!
_class
loc:@dense_1/kernel*
dtype0*
value	B :*
_output_shapes
: 
?
Ldense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV29dense_1/kernel/Initializer/stateless_random_uniform/shapePdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterRdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1Pdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
Tshape0*!
_class
loc:@dense_1/kernel*
_output_shapes

:

?
7dense_1/kernel/Initializer/stateless_random_uniform/subSub7dense_1/kernel/Initializer/stateless_random_uniform/max7dense_1/kernel/Initializer/stateless_random_uniform/min*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
: 
?
7dense_1/kernel/Initializer/stateless_random_uniform/mulMulLdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV27dense_1/kernel/Initializer/stateless_random_uniform/sub*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:

?
3dense_1/kernel/Initializer/stateless_random_uniformAddV27dense_1/kernel/Initializer/stateless_random_uniform/mul7dense_1/kernel/Initializer/stateless_random_uniform/min*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:

?
dense_1/kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
shape
:
*
shared_namedense_1/kernel
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
?
dense_1/kernel/AssignAssignVariableOpdense_1/kernel3dense_1/kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

?
dense_1/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_1/bias*
valueB*    *
_output_shapes
:
?
dense_1/biasVarHandleOp*
	container *
allowed_devices
 *
_class
loc:@dense_1/bias*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
x
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0*
validate_shape( 
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

?
dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
transpose_b( *
T0*
transpose_a( *
_output_shapes

:
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
?
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*
_output_shapes

:
T
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*
T0*
_output_shapes

:ݪ
?
?
$batch_normalization_2_cond_false_3028
*readvariableop_batch_normalization_2_gamma:9
+readvariableop_1_batch_normalization_2_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:8
4fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
epsilon%??'7*B
_output_shapes0
.:9:::::*
is_training( *
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :9:,(
&
_output_shapes
:9
?
3
&batch_normalization_4_cond_1_false_629	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
2
%batch_normalization_1_cond_1_true_240	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
0
#batch_normalization_cond_1_true_111	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
$batch_normalization_4_cond_false_5618
*readvariableop_batch_normalization_4_gamma:9
+readvariableop_1_batch_normalization_4_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:8
4fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
epsilon%??'7*B
_output_shapes0
.::::::*
is_training( *
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
3
&batch_normalization_1_cond_1_false_241	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
2
%batch_normalization_3_cond_1_true_499	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
2
%batch_normalization_4_cond_1_true_628	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
#batch_normalization_4_cond_true_5608
*readvariableop_batch_normalization_4_gamma:9
+readvariableop_1_batch_normalization_4_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:8
4fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
epsilon%??'7*B
_output_shapes0
.::::::*
is_training(*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
2
%batch_normalization_2_cond_1_true_369	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
!batch_normalization_cond_false_446
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:*
&fusedbatchnormv3_stream_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?s
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
dtype0*
_output_shapes
:v
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV3&fusedbatchnormv3_stream_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.:{:::::*
is_training( *
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :{:,(
&
_output_shapes
:{
?
?
#batch_normalization_3_cond_true_4318
*readvariableop_batch_normalization_3_gamma:9
+readvariableop_1_batch_normalization_3_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:8
4fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.::::::*
is_training(*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
3
&batch_normalization_3_cond_1_false_500	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
$batch_normalization_3_cond_false_4328
*readvariableop_batch_normalization_3_gamma:9
+readvariableop_1_batch_normalization_3_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:8
4fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
epsilon%??'7*B
_output_shapes0
.::::::*
is_training( *
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
1
$batch_normalization_cond_1_false_112	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
#batch_normalization_1_cond_true_1728
*readvariableop_batch_normalization_1_gamma:9
+readvariableop_1_batch_normalization_1_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance:6
2fusedbatchnormv3_stream_1_depthwise_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV32fusedbatchnormv3_stream_1_depthwise_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
epsilon%??'7*B
_output_shapes0
.:<:::::*
is_training(*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :<:,(
&
_output_shapes
:<
?
?
dropout_cond_false_650
identity_activation_4_relu
identity
optionalnone
optionalnone_1
optionalnone_2
optionalnone_3
optionalnone_4
optionalnone_5
optionalnone_6a
IdentityIdentityidentity_activation_4_relu*
T0*&
_output_shapes
:4
OptionalNoneOptionalNone*
_output_shapes
: 6
OptionalNone_1OptionalNone*
_output_shapes
: 6
OptionalNone_2OptionalNone*
_output_shapes
: 6
OptionalNone_3OptionalNone*
_output_shapes
: 6
OptionalNone_4OptionalNone*
_output_shapes
: 6
OptionalNone_5OptionalNone*
_output_shapes
: 6
OptionalNone_6OptionalNone*
_output_shapes
: "+
optionalnone_3OptionalNone_3:optional:0"+
optionalnone_2OptionalNone_2:optional:0"+
optionalnone_6OptionalNone_6:optional:0"'
optionalnoneOptionalNone:optional:0"+
optionalnone_4OptionalNone_4:optional:0"+
optionalnone_5OptionalNone_5:optional:0"
identityIdentity:output:0"+
optionalnone_1OptionalNone_1:optional:0*%
_input_shapes
::, (
&
_output_shapes
:
?
?
dropout_cond_true_649!
dropout_mul_activation_4_relu
dropout_selectv2
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?R
dropout/ConstConst*
dtype0*
valueB
 *   @*
_output_shapes
: z
dropout/MulMuldropout_mul_activation_4_reludropout/Const:output:0*
T0*&
_output_shapes
:f
dropout/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*

seed *
T0*
seed2 *&
_output_shapes
:[
dropout/GreaterEqual/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:T
dropout/Const_1Const*
dtype0*
valueB
 *    *
_output_shapes
: ?
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*&
_output_shapes
:?
OptionalFromValueOptionalFromValuedropout/Const:output:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValuedropout/Mul:z:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValuedropout/Shape:output:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue-dropout/random_uniform/RandomUniform:output:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValuedropout/GreaterEqual/y:output:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValuedropout/GreaterEqual:z:0*
Toutput_types
2
*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValuedropout/Const_1:output:0*
Toutput_types
2*
_output_shapes
: :???"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"-
dropout_selectv2dropout/SelectV2:output:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*%
_input_shapes
::, (
&
_output_shapes
:
?
?
 batch_normalization_cond_true_436
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:*
&fusedbatchnormv3_stream_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?s
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
dtype0*
_output_shapes
:v
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV3&fusedbatchnormv3_stream_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.:{:::::*
is_training(*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :{:,(
&
_output_shapes
:{
?
3
&batch_normalization_2_cond_1_false_370	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
#batch_normalization_2_cond_true_3018
*readvariableop_batch_normalization_2_gamma:9
+readvariableop_1_batch_normalization_2_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:8
4fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
epsilon%??'7*B
_output_shapes0
.:9:::::*
is_training(*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :9:,(
&
_output_shapes
:9
?
?
$batch_normalization_1_cond_false_1738
*readvariableop_batch_normalization_1_gamma:9
+readvariableop_1_batch_normalization_1_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance:6
2fusedbatchnormv3_stream_1_depthwise_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV32fusedbatchnormv3_stream_1_depthwise_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.:<:::::*
is_training( *
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :<:,(
&
_output_shapes
:<"?
Z?=?     
v?@		?6j???AJ??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
E
AssignSubVariableOp
resource
value"dtype"
dtypetype?
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
?
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 ?
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
^
StatelessRandomGetKeyCounter
seed"Tseed
key
counter"
Tseedtype0	:
2	
?
StatelessRandomUniformV2
shape"Tshape
key
counter
alg
output"dtype"
dtypetype0:
2"
Tshapetype0:
2	
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
9
VarIsInitializedOp
resource
is_initialized
?*2.12.0-dev202302032v1.12.1-88849-g3e5b6f1899f??
j
input_1Placeholder*
dtype0*
shape:?	*'
_output_shapes
:?	
?
max_pooling2d/MaxPoolMaxPoolinput_1*
explicit_paddings
 *
T0*
ksize
*'
_output_shapes
:?*
strides
*
data_formatNHWC*
paddingVALID
?
8stream/kernel/Initializer/stateless_random_uniform/shapeConst*
dtype0* 
_class
loc:@stream/kernel*%
valueB"            *
_output_shapes
:
?
6stream/kernel/Initializer/stateless_random_uniform/minConst*
dtype0* 
_class
loc:@stream/kernel*
valueB
 *   ?*
_output_shapes
: 
?
6stream/kernel/Initializer/stateless_random_uniform/maxConst* 
_class
loc:@stream/kernel*
dtype0*
valueB
 *   ?*
_output_shapes
: 
?
Tstream/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
dtype0* 
_class
loc:@stream/kernel*
valueB"??0    *
_output_shapes
:
?
Ostream/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterTstream/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed* 
_class
loc:@stream/kernel*
Tseed0* 
_output_shapes
::
?
Ostream/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst* 
_class
loc:@stream/kernel*
dtype0*
value	B :*
_output_shapes
: 
?
Kstream/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV28stream/kernel/Initializer/stateless_random_uniform/shapeOstream/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterQstream/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1Ostream/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
Tshape0* 
_class
loc:@stream/kernel*&
_output_shapes
:
?
6stream/kernel/Initializer/stateless_random_uniform/subSub6stream/kernel/Initializer/stateless_random_uniform/max6stream/kernel/Initializer/stateless_random_uniform/min* 
_class
loc:@stream/kernel*
T0*
_output_shapes
: 
?
6stream/kernel/Initializer/stateless_random_uniform/mulMulKstream/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV26stream/kernel/Initializer/stateless_random_uniform/sub* 
_class
loc:@stream/kernel*
T0*&
_output_shapes
:
?
2stream/kernel/Initializer/stateless_random_uniformAddV26stream/kernel/Initializer/stateless_random_uniform/mul6stream/kernel/Initializer/stateless_random_uniform/min* 
_class
loc:@stream/kernel*
T0*&
_output_shapes
:
?
stream/kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: * 
_class
loc:@stream/kernel*
shape:*
shared_namestream/kernel
k
.stream/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream/kernel*
_output_shapes
: 
?
stream/kernel/AssignAssignVariableOpstream/kernel2stream/kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
w
!stream/kernel/Read/ReadVariableOpReadVariableOpstream/kernel*
dtype0*&
_output_shapes
:
?
stream/bias/Initializer/zerosConst*
dtype0*
_class
loc:@stream/bias*
valueB*    *
_output_shapes
:
?
stream/biasVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *
_class
loc:@stream/bias*
shape:*
shared_namestream/bias
g
,stream/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream/bias*
_output_shapes
: 
u
stream/bias/AssignAssignVariableOpstream/biasstream/bias/Initializer/zeros*
dtype0*
validate_shape( 
g
stream/bias/Read/ReadVariableOpReadVariableOpstream/bias*
dtype0*
_output_shapes
:
y
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOpstream/kernel*
dtype0*&
_output_shapes
:
?
stream/conv2d/Conv2DConv2Dmax_pooling2d/MaxPool#stream/conv2d/Conv2D/ReadVariableOp*
	dilations
*
explicit_paddings
 *
T0*
use_cudnn_on_gpu(*&
_output_shapes
:{*
strides
*
data_formatNHWC*
paddingVALID
l
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOpstream/bias*
dtype0*
_output_shapes
:
?
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D$stream/conv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:{
?
*batch_normalization/gamma/Initializer/onesConst*
dtype0*,
_class"
 loc:@batch_normalization/gamma*
valueB*  ??*
_output_shapes
:
?
batch_normalization/gammaVarHandleOp*
	container *
allowed_devices
 *,
_class"
 loc:@batch_normalization/gamma*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
?
:batch_normalization/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/gamma*
_output_shapes
: 
?
 batch_normalization/gamma/AssignAssignVariableOpbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
dtype0*
_output_shapes
:
?
*batch_normalization/beta/Initializer/zerosConst*+
_class!
loc:@batch_normalization/beta*
dtype0*
valueB*    *
_output_shapes
:
?
batch_normalization/betaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *+
_class!
loc:@batch_normalization/beta*
shape:*)
shared_namebatch_normalization/beta
?
9batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
?
batch_normalization/beta/AssignAssignVariableOpbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
dtype0*
_output_shapes
:
?
1batch_normalization/moving_mean/Initializer/zerosConst*
dtype0*2
_class(
&$loc:@batch_normalization/moving_mean*
valueB*    *
_output_shapes
:
?
batch_normalization/moving_meanVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *2
_class(
&$loc:@batch_normalization/moving_mean*
shape:*0
shared_name!batch_normalization/moving_mean
?
@batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
?
&batch_normalization/moving_mean/AssignAssignVariableOpbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
:
?
4batch_normalization/moving_variance/Initializer/onesConst*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
valueB*  ??*
_output_shapes
:
?
#batch_normalization/moving_varianceVarHandleOp*
	container *
allowed_devices
 *6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
?
Dbatch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
?
*batch_normalization/moving_variance/AssignAssignVariableOp#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
:
\
keras_learning_phase/inputConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
dtype0
*
shape: *
_output_shapes
: 
?
batch_normalization/condIfkeras_learning_phasebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancestream/conv2d/BiasAdd*&
_read_only_resource_inputs
*4
else_branch%R#
!batch_normalization_cond_false_44*3
then_branch$R"
 batch_normalization_cond_true_43*
Tcond0
*@
_output_shapes.
,:{::: : : : : : : *
_lower_using_switch_merge(*?
output_shapes.
,:{::: : : : : : : *
Tout
2
*
Tin	
2
x
!batch_normalization/cond/IdentityIdentitybatch_normalization/cond*
T0*&
_output_shapes
:{
p
#batch_normalization/cond/Identity_1Identitybatch_normalization/cond:1*
T0*
_output_shapes
:
p
#batch_normalization/cond/Identity_2Identitybatch_normalization/cond:2*
T0*
_output_shapes
:
l
#batch_normalization/cond/Identity_3Identitybatch_normalization/cond:3*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_4Identitybatch_normalization/cond:4*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_5Identitybatch_normalization/cond:5*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_6Identitybatch_normalization/cond:6*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_7Identitybatch_normalization/cond:7*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_8Identitybatch_normalization/cond:8*
T0*
_output_shapes
: 
l
#batch_normalization/cond/Identity_9Identitybatch_normalization/cond:9*
T0*
_output_shapes
: 
?
batch_normalization/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *7
else_branch(R&
$batch_normalization_cond_1_false_112*6
then_branch'R%
#batch_normalization_cond_1_true_111*
Tcond0
*
_output_shapes
: *
_lower_using_switch_merge(*
output_shapes
: *
Tout
2*	
Tin
 
l
#batch_normalization/cond_1/IdentityIdentitybatch_normalization/cond_1*
T0*
_output_shapes
: 
?
)batch_normalization/AssignMovingAvg/sub/xConst*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
'batch_normalization/AssignMovingAvg/subSub)batch_normalization/AssignMovingAvg/sub/x#batch_normalization/cond_1/Identity*2
_class(
&$loc:@batch_normalization/moving_mean*
T0*
_output_shapes
: 
?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
:
?
)batch_normalization/AssignMovingAvg/sub_1Sub2batch_normalization/AssignMovingAvg/ReadVariableOp#batch_normalization/cond/Identity_1*2
_class(
&$loc:@batch_normalization/moving_mean*
T0*
_output_shapes
:
?
'batch_normalization/AssignMovingAvg/mulMul)batch_normalization/AssignMovingAvg/sub_1'batch_normalization/AssignMovingAvg/sub*2
_class(
&$loc:@batch_normalization/moving_mean*
T0*
_output_shapes
:
?
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/mul*2
_class(
&$loc:@batch_normalization/moving_mean*&
 _has_manual_control_dependencies(*
dtype0
?
4batch_normalization/AssignMovingAvg/ReadVariableOp_1ReadVariableOpbatch_normalization/moving_mean8^batch_normalization/AssignMovingAvg/AssignSubVariableOp*
dtype0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:
?
+batch_normalization/AssignMovingAvg_1/sub/xConst*
dtype0*6
_class,
*(loc:@batch_normalization/moving_variance*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization/AssignMovingAvg_1/subSub+batch_normalization/AssignMovingAvg_1/sub/x#batch_normalization/cond_1/Identity*6
_class,
*(loc:@batch_normalization/moving_variance*
T0*
_output_shapes
: 
?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
:
?
+batch_normalization/AssignMovingAvg_1/sub_1Sub4batch_normalization/AssignMovingAvg_1/ReadVariableOp#batch_normalization/cond/Identity_2*6
_class,
*(loc:@batch_normalization/moving_variance*
T0*
_output_shapes
:
?
)batch_normalization/AssignMovingAvg_1/mulMul+batch_normalization/AssignMovingAvg_1/sub_1)batch_normalization/AssignMovingAvg_1/sub*6
_class,
*(loc:@batch_normalization/moving_variance*
T0*
_output_shapes
:
?
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/mul*6
_class,
*(loc:@batch_normalization/moving_variance*&
 _has_manual_control_dependencies(*
dtype0
?
6batch_normalization/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp#batch_normalization/moving_variance:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
:
k
activation/ReluRelu!batch_normalization/cond/Identity*
T0*&
_output_shapes
:{
?
Dstream_1/depthwise_kernel/Initializer/stateless_random_uniform/shapeConst*,
_class"
 loc:@stream_1/depthwise_kernel*
dtype0*%
valueB"            *
_output_shapes
:
?
Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/minConst*,
_class"
 loc:@stream_1/depthwise_kernel*
dtype0*
valueB
 *?7?*
_output_shapes
: 
?
Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/maxConst*,
_class"
 loc:@stream_1/depthwise_kernel*
dtype0*
valueB
 *?7?*
_output_shapes
: 
?
`stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*,
_class"
 loc:@stream_1/depthwise_kernel*
dtype0*
valueB"P?\    *
_output_shapes
:
?
[stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter`stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*,
_class"
 loc:@stream_1/depthwise_kernel*
Tseed0* 
_output_shapes
::
?
[stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*
dtype0*,
_class"
 loc:@stream_1/depthwise_kernel*
value	B :*
_output_shapes
: 
?
Wstream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Dstream_1/depthwise_kernel/Initializer/stateless_random_uniform/shape[stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter]stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1[stream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*,
_class"
 loc:@stream_1/depthwise_kernel*
Tshape0*
dtype0*&
_output_shapes
:
?
Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/subSubBstream_1/depthwise_kernel/Initializer/stateless_random_uniform/maxBstream_1/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_1/depthwise_kernel*
T0*
_output_shapes
: 
?
Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/mulMulWstream_1/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/sub*,
_class"
 loc:@stream_1/depthwise_kernel*
T0*&
_output_shapes
:
?
>stream_1/depthwise_kernel/Initializer/stateless_random_uniformAddV2Bstream_1/depthwise_kernel/Initializer/stateless_random_uniform/mulBstream_1/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_1/depthwise_kernel*
T0*&
_output_shapes
:
?
stream_1/depthwise_kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *,
_class"
 loc:@stream_1/depthwise_kernel*
shape:**
shared_namestream_1/depthwise_kernel
?
:stream_1/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_1/depthwise_kernel*
_output_shapes
: 
?
 stream_1/depthwise_kernel/AssignAssignVariableOpstream_1/depthwise_kernel>stream_1/depthwise_kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
?
-stream_1/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_1/depthwise_kernel*
dtype0*&
_output_shapes
:
?
stream_1/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@stream_1/bias*
valueB*    *
_output_shapes
:
?
stream_1/biasVarHandleOp*
	container *
allowed_devices
 * 
_class
loc:@stream_1/bias*
_output_shapes
: *
dtype0*
shape:*
shared_namestream_1/bias
k
.stream_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_1/bias*
_output_shapes
: 
{
stream_1/bias/AssignAssignVariableOpstream_1/biasstream_1/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_1/bias/Read/ReadVariableOpReadVariableOpstream_1/bias*
dtype0*
_output_shapes
:
?
2stream_1/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpstream_1/depthwise_kernel*
dtype0*&
_output_shapes
:
?
)stream_1/depthwise_conv2d/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
1stream_1/depthwise_conv2d/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
#stream_1/depthwise_conv2d/depthwiseDepthwiseConv2dNativeactivation/Relu2stream_1/depthwise_conv2d/depthwise/ReadVariableOp*
	dilations
*
explicit_paddings
 *
T0*&
_output_shapes
:<*
strides
*
data_formatNHWC*
paddingVALID
z
0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpstream_1/bias*
dtype0*
_output_shapes
:
?
!stream_1/depthwise_conv2d/BiasAddBiasAdd#stream_1/depthwise_conv2d/depthwise0stream_1/depthwise_conv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:<
?
,batch_normalization_1/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
valueB*  ??*
_output_shapes
:
?
batch_normalization_1/gammaVarHandleOp*
	container *
allowed_devices
 *.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
?
<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
?
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_1/beta/Initializer/zerosConst*
dtype0*-
_class#
!loc:@batch_normalization_1/beta*
valueB*    *
_output_shapes
:
?
batch_normalization_1/betaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_1/beta*
shape:*+
shared_namebatch_normalization_1/beta
?
;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
?
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_1/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
valueB*    *
_output_shapes
:
?
!batch_normalization_1/moving_meanVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_1/moving_mean*
shape:*2
shared_name#!batch_normalization_1/moving_mean
?
Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
?
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_1/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
valueB*  ??*
_output_shapes
:
?
%batch_normalization_1/moving_varianceVarHandleOp*
	container *
allowed_devices
 *8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
?
Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
?
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_1/condIfkeras_learning_phasebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!stream_1/depthwise_conv2d/BiasAdd*&
_read_only_resource_inputs
*?
output_shapes.
,:<::: : : : : : : *6
then_branch'R%
#batch_normalization_1_cond_true_172*
Tcond0
*@
_output_shapes.
,:<::: : : : : : : *
_lower_using_switch_merge(*7
else_branch(R&
$batch_normalization_1_cond_false_173*
Tout
2
*
Tin	
2
|
#batch_normalization_1/cond/IdentityIdentitybatch_normalization_1/cond*
T0*&
_output_shapes
:<
t
%batch_normalization_1/cond/Identity_1Identitybatch_normalization_1/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_1/cond/Identity_2Identitybatch_normalization_1/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_1/cond/Identity_3Identitybatch_normalization_1/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_4Identitybatch_normalization_1/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_5Identitybatch_normalization_1/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_6Identitybatch_normalization_1/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_7Identitybatch_normalization_1/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_8Identitybatch_normalization_1/cond:8*
T0*
_output_shapes
: 
p
%batch_normalization_1/cond/Identity_9Identitybatch_normalization_1/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_1/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *
output_shapes
: *8
then_branch)R'
%batch_normalization_1_cond_1_true_240*
Tcond0
*
_output_shapes
: *
_lower_using_switch_merge(*9
else_branch*R(
&batch_normalization_1_cond_1_false_241*
Tout
2*	
Tin
 
p
%batch_normalization_1/cond_1/IdentityIdentitybatch_normalization_1/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_1/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization_1/AssignMovingAvg/subSub+batch_normalization_1/AssignMovingAvg/sub/x%batch_normalization_1/cond_1/Identity*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_1/AssignMovingAvg/sub_1Sub4batch_normalization_1/AssignMovingAvg/ReadVariableOp%batch_normalization_1/cond/Identity_1*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0*
_output_shapes
:
?
)batch_normalization_1/AssignMovingAvg/mulMul+batch_normalization_1/AssignMovingAvg/sub_1)batch_normalization_1/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0*
_output_shapes
:
?
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*
dtype0*&
 _has_manual_control_dependencies(*4
_class*
(&loc:@batch_normalization_1/moving_mean
?
6batch_normalization_1/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp*
dtype0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:
?
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
+batch_normalization_1/AssignMovingAvg_1/subSub-batch_normalization_1/AssignMovingAvg_1/sub/x%batch_normalization_1/cond_1/Identity*8
_class.
,*loc:@batch_normalization_1/moving_variance*
T0*
_output_shapes
: 
?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%batch_normalization_1/cond/Identity_2*8
_class.
,*loc:@batch_normalization_1/moving_variance*
T0*
_output_shapes
:
?
+batch_normalization_1/AssignMovingAvg_1/mulMul-batch_normalization_1/AssignMovingAvg_1/sub_1+batch_normalization_1/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_1/moving_variance*
T0*
_output_shapes
:
?
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_1/moving_variance*&
 _has_manual_control_dependencies(*
dtype0
?
8batch_normalization_1/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:
o
activation_1/ReluRelu#batch_normalization_1/cond/Identity*
T0*&
_output_shapes
:<
?
Dstream_2/depthwise_kernel/Initializer/stateless_random_uniform/shapeConst*,
_class"
 loc:@stream_2/depthwise_kernel*
dtype0*%
valueB"            *
_output_shapes
:
?
Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/minConst*,
_class"
 loc:@stream_2/depthwise_kernel*
dtype0*
valueB
 *?7?*
_output_shapes
: 
?
Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/maxConst*
dtype0*,
_class"
 loc:@stream_2/depthwise_kernel*
valueB
 *?7?*
_output_shapes
: 
?
`stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
dtype0*,
_class"
 loc:@stream_2/depthwise_kernel*
valueB"?.o    *
_output_shapes
:
?
[stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter`stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*,
_class"
 loc:@stream_2/depthwise_kernel*
Tseed0* 
_output_shapes
::
?
[stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*
dtype0*,
_class"
 loc:@stream_2/depthwise_kernel*
value	B :*
_output_shapes
: 
?
Wstream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Dstream_2/depthwise_kernel/Initializer/stateless_random_uniform/shape[stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter]stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1[stream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
Tshape0*,
_class"
 loc:@stream_2/depthwise_kernel*&
_output_shapes
:
?
Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/subSubBstream_2/depthwise_kernel/Initializer/stateless_random_uniform/maxBstream_2/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_2/depthwise_kernel*
T0*
_output_shapes
: 
?
Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/mulMulWstream_2/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/sub*,
_class"
 loc:@stream_2/depthwise_kernel*
T0*&
_output_shapes
:
?
>stream_2/depthwise_kernel/Initializer/stateless_random_uniformAddV2Bstream_2/depthwise_kernel/Initializer/stateless_random_uniform/mulBstream_2/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_2/depthwise_kernel*
T0*&
_output_shapes
:
?
stream_2/depthwise_kernelVarHandleOp*
	container *
allowed_devices
 *,
_class"
 loc:@stream_2/depthwise_kernel*
_output_shapes
: *
dtype0*
shape:**
shared_namestream_2/depthwise_kernel
?
:stream_2/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_2/depthwise_kernel*
_output_shapes
: 
?
 stream_2/depthwise_kernel/AssignAssignVariableOpstream_2/depthwise_kernel>stream_2/depthwise_kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
?
-stream_2/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_2/depthwise_kernel*
dtype0*&
_output_shapes
:
?
stream_2/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@stream_2/bias*
valueB*    *
_output_shapes
:
?
stream_2/biasVarHandleOp*
	container *
allowed_devices
 * 
_class
loc:@stream_2/bias*
_output_shapes
: *
dtype0*
shape:*
shared_namestream_2/bias
k
.stream_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_2/bias*
_output_shapes
: 
{
stream_2/bias/AssignAssignVariableOpstream_2/biasstream_2/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_2/bias/Read/ReadVariableOpReadVariableOpstream_2/bias*
dtype0*
_output_shapes
:
?
4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpstream_2/depthwise_kernel*
dtype0*&
_output_shapes
:
?
+stream_2/depthwise_conv2d_1/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
3stream_2/depthwise_conv2d_1/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
%stream_2/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativeactivation_1/Relu4stream_2/depthwise_conv2d_1/depthwise/ReadVariableOp*
	dilations
*
explicit_paddings
 *
T0*&
_output_shapes
:9*
strides
*
data_formatNHWC*
paddingVALID
|
2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpstream_2/bias*
dtype0*
_output_shapes
:
?
#stream_2/depthwise_conv2d_1/BiasAddBiasAdd%stream_2/depthwise_conv2d_1/depthwise2stream_2/depthwise_conv2d_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:9
?
,batch_normalization_2/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
valueB*  ??*
_output_shapes
:
?
batch_normalization_2/gammaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_2/gamma*
shape:*,
shared_namebatch_normalization_2/gamma
?
<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
?
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_2/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
valueB*    *
_output_shapes
:
?
batch_normalization_2/betaVarHandleOp*
	container *
allowed_devices
 *-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_2/beta
?
;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
?
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_2/moving_mean/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@batch_normalization_2/moving_mean*
valueB*    *
_output_shapes
:
?
!batch_normalization_2/moving_meanVarHandleOp*
	container *
allowed_devices
 *4
_class*
(&loc:@batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_2/moving_mean
?
Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
?
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_2/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
valueB*  ??*
_output_shapes
:
?
%batch_normalization_2/moving_varianceVarHandleOp*
	container *
allowed_devices
 *8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_2/moving_variance
?
Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
?
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_2/condIfkeras_learning_phasebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#stream_2/depthwise_conv2d_1/BiasAdd*&
_read_only_resource_inputs
*7
else_branch(R&
$batch_normalization_2_cond_false_302*6
then_branch'R%
#batch_normalization_2_cond_true_301*
Tcond0
*@
_output_shapes.
,:9::: : : : : : : *
_lower_using_switch_merge(*?
output_shapes.
,:9::: : : : : : : *
Tout
2
*
Tin	
2
|
#batch_normalization_2/cond/IdentityIdentitybatch_normalization_2/cond*
T0*&
_output_shapes
:9
t
%batch_normalization_2/cond/Identity_1Identitybatch_normalization_2/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_2/cond/Identity_2Identitybatch_normalization_2/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_2/cond/Identity_3Identitybatch_normalization_2/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_4Identitybatch_normalization_2/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_5Identitybatch_normalization_2/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_6Identitybatch_normalization_2/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_7Identitybatch_normalization_2/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_8Identitybatch_normalization_2/cond:8*
T0*
_output_shapes
: 
p
%batch_normalization_2/cond/Identity_9Identitybatch_normalization_2/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_2/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *9
else_branch*R(
&batch_normalization_2_cond_1_false_370*8
then_branch)R'
%batch_normalization_2_cond_1_true_369*
Tcond0
*
_output_shapes
: *
_lower_using_switch_merge(*
output_shapes
: *
Tout
2*	
Tin
 
p
%batch_normalization_2/cond_1/IdentityIdentitybatch_normalization_2/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_2/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization_2/AssignMovingAvg/subSub+batch_normalization_2/AssignMovingAvg/sub/x%batch_normalization_2/cond_1/Identity*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_2/AssignMovingAvg/sub_1Sub4batch_normalization_2/AssignMovingAvg/ReadVariableOp%batch_normalization_2/cond/Identity_1*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0*
_output_shapes
:
?
)batch_normalization_2/AssignMovingAvg/mulMul+batch_normalization_2/AssignMovingAvg/sub_1)batch_normalization_2/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0*
_output_shapes
:
?
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*
dtype0*&
 _has_manual_control_dependencies(*4
_class*
(&loc:@batch_normalization_2/moving_mean
?
6batch_normalization_2/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_2/moving_mean:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:
?
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
+batch_normalization_2/AssignMovingAvg_1/subSub-batch_normalization_2/AssignMovingAvg_1/sub/x%batch_normalization_2/cond_1/Identity*8
_class.
,*loc:@batch_normalization_2/moving_variance*
T0*
_output_shapes
: 
?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%batch_normalization_2/cond/Identity_2*8
_class.
,*loc:@batch_normalization_2/moving_variance*
T0*
_output_shapes
:
?
+batch_normalization_2/AssignMovingAvg_1/mulMul-batch_normalization_2/AssignMovingAvg_1/sub_1+batch_normalization_2/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_2/moving_variance*
T0*
_output_shapes
:
?
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_2/moving_variance*&
 _has_manual_control_dependencies(*
dtype0
?
8batch_normalization_2/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_2/moving_variance<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:
o
activation_2/ReluRelu#batch_normalization_2/cond/Identity*
T0*&
_output_shapes
:9
?
"stream_3/average_pooling2d/AvgPoolAvgPoolactivation_2/Relu*
strides
*
ksize
*
T0*
data_formatNHWC*
paddingVALID*&
_output_shapes
:
?
Dstream_4/depthwise_kernel/Initializer/stateless_random_uniform/shapeConst*,
_class"
 loc:@stream_4/depthwise_kernel*
dtype0*%
valueB"            *
_output_shapes
:
?
Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/minConst*
dtype0*,
_class"
 loc:@stream_4/depthwise_kernel*
valueB
 *׳ݾ*
_output_shapes
: 
?
Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/maxConst*,
_class"
 loc:@stream_4/depthwise_kernel*
dtype0*
valueB
 *׳?>*
_output_shapes
: 
?
`stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
dtype0*,
_class"
 loc:@stream_4/depthwise_kernel*
valueB"t'I    *
_output_shapes
:
?
[stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter`stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*,
_class"
 loc:@stream_4/depthwise_kernel*
Tseed0* 
_output_shapes
::
?
[stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*
dtype0*,
_class"
 loc:@stream_4/depthwise_kernel*
value	B :*
_output_shapes
: 
?
Wstream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Dstream_4/depthwise_kernel/Initializer/stateless_random_uniform/shape[stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter]stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1[stream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*,
_class"
 loc:@stream_4/depthwise_kernel*
Tshape0*
dtype0*&
_output_shapes
:
?
Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/subSubBstream_4/depthwise_kernel/Initializer/stateless_random_uniform/maxBstream_4/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_4/depthwise_kernel*
T0*
_output_shapes
: 
?
Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/mulMulWstream_4/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/sub*,
_class"
 loc:@stream_4/depthwise_kernel*
T0*&
_output_shapes
:
?
>stream_4/depthwise_kernel/Initializer/stateless_random_uniformAddV2Bstream_4/depthwise_kernel/Initializer/stateless_random_uniform/mulBstream_4/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_4/depthwise_kernel*
T0*&
_output_shapes
:
?
stream_4/depthwise_kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *,
_class"
 loc:@stream_4/depthwise_kernel*
shape:**
shared_namestream_4/depthwise_kernel
?
:stream_4/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_4/depthwise_kernel*
_output_shapes
: 
?
 stream_4/depthwise_kernel/AssignAssignVariableOpstream_4/depthwise_kernel>stream_4/depthwise_kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
?
-stream_4/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_4/depthwise_kernel*
dtype0*&
_output_shapes
:
?
stream_4/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@stream_4/bias*
valueB*    *
_output_shapes
:
?
stream_4/biasVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: * 
_class
loc:@stream_4/bias*
shape:*
shared_namestream_4/bias
k
.stream_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_4/bias*
_output_shapes
: 
{
stream_4/bias/AssignAssignVariableOpstream_4/biasstream_4/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_4/bias/Read/ReadVariableOpReadVariableOpstream_4/bias*
dtype0*
_output_shapes
:
?
4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpstream_4/depthwise_kernel*
dtype0*&
_output_shapes
:
?
+stream_4/depthwise_conv2d_2/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
3stream_4/depthwise_conv2d_2/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
%stream_4/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative"stream_3/average_pooling2d/AvgPool4stream_4/depthwise_conv2d_2/depthwise/ReadVariableOp*
	dilations
*
explicit_paddings
 *
T0*&
_output_shapes
:*
strides
*
data_formatNHWC*
paddingVALID
|
2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpstream_4/bias*
dtype0*
_output_shapes
:
?
#stream_4/depthwise_conv2d_2/BiasAddBiasAdd%stream_4/depthwise_conv2d_2/depthwise2stream_4/depthwise_conv2d_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:
?
,batch_normalization_3/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0*
valueB*  ??*
_output_shapes
:
?
batch_normalization_3/gammaVarHandleOp*
	container *
allowed_devices
 *.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma
?
<batch_normalization_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
?
"batch_normalization_3/gamma/AssignAssignVariableOpbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_3/beta/Initializer/zerosConst*
dtype0*-
_class#
!loc:@batch_normalization_3/beta*
valueB*    *
_output_shapes
:
?
batch_normalization_3/betaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *-
_class#
!loc:@batch_normalization_3/beta*
shape:*+
shared_namebatch_normalization_3/beta
?
;batch_normalization_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
?
!batch_normalization_3/beta/AssignAssignVariableOpbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_3/moving_mean/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB*    *
_output_shapes
:
?
!batch_normalization_3/moving_meanVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_3/moving_mean*
shape:*2
shared_name#!batch_normalization_3/moving_mean
?
Bbatch_normalization_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
?
(batch_normalization_3/moving_mean/AssignAssignVariableOp!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_3/moving_variance/Initializer/onesConst*
dtype0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB*  ??*
_output_shapes
:
?
%batch_normalization_3/moving_varianceVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_3/moving_variance*
shape:*6
shared_name'%batch_normalization_3/moving_variance
?
Fbatch_normalization_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
?
,batch_normalization_3/moving_variance/AssignAssignVariableOp%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_3/condIfkeras_learning_phasebatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#stream_4/depthwise_conv2d_2/BiasAdd*&
_read_only_resource_inputs
*7
else_branch(R&
$batch_normalization_3_cond_false_432*6
then_branch'R%
#batch_normalization_3_cond_true_431*
Tcond0
*@
_output_shapes.
,:::: : : : : : : *
_lower_using_switch_merge(*?
output_shapes.
,:::: : : : : : : *
Tout
2
*
Tin	
2
|
#batch_normalization_3/cond/IdentityIdentitybatch_normalization_3/cond*
T0*&
_output_shapes
:
t
%batch_normalization_3/cond/Identity_1Identitybatch_normalization_3/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_3/cond/Identity_2Identitybatch_normalization_3/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_3/cond/Identity_3Identitybatch_normalization_3/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_4Identitybatch_normalization_3/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_5Identitybatch_normalization_3/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_6Identitybatch_normalization_3/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_7Identitybatch_normalization_3/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_8Identitybatch_normalization_3/cond:8*
T0*
_output_shapes
: 
p
%batch_normalization_3/cond/Identity_9Identitybatch_normalization_3/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_3/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *9
else_branch*R(
&batch_normalization_3_cond_1_false_500*8
then_branch)R'
%batch_normalization_3_cond_1_true_499*
Tcond0
*
_output_shapes
: *
_lower_using_switch_merge(*
output_shapes
: *
Tout
2*	
Tin
 
p
%batch_normalization_3/cond_1/IdentityIdentitybatch_normalization_3/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_3/AssignMovingAvg/sub/xConst*
dtype0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization_3/AssignMovingAvg/subSub+batch_normalization_3/AssignMovingAvg/sub/x%batch_normalization_3/cond_1/Identity*4
_class*
(&loc:@batch_normalization_3/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_3/AssignMovingAvg/sub_1Sub4batch_normalization_3/AssignMovingAvg/ReadVariableOp%batch_normalization_3/cond/Identity_1*4
_class*
(&loc:@batch_normalization_3/moving_mean*
T0*
_output_shapes
:
?
)batch_normalization_3/AssignMovingAvg/mulMul+batch_normalization_3/AssignMovingAvg/sub_1)batch_normalization_3/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_3/moving_mean*
T0*
_output_shapes
:
?
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*
dtype0*&
 _has_manual_control_dependencies(*4
_class*
(&loc:@batch_normalization_3/moving_mean
?
6batch_normalization_3/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:
?
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*
dtype0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
valueB
 *  ??*
_output_shapes
: 
?
+batch_normalization_3/AssignMovingAvg_1/subSub-batch_normalization_3/AssignMovingAvg_1/sub/x%batch_normalization_3/cond_1/Identity*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0*
_output_shapes
: 
?
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp%batch_normalization_3/cond/Identity_2*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0*
_output_shapes
:
?
+batch_normalization_3/AssignMovingAvg_1/mulMul-batch_normalization_3/AssignMovingAvg_1/sub_1+batch_normalization_3/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0*
_output_shapes
:
?
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_3/moving_variance*&
 _has_manual_control_dependencies(*
dtype0
?
8batch_normalization_3/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_3/moving_variance<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:
o
activation_3/ReluRelu#batch_normalization_3/cond/Identity*
T0*&
_output_shapes
:
?
Dstream_5/depthwise_kernel/Initializer/stateless_random_uniform/shapeConst*,
_class"
 loc:@stream_5/depthwise_kernel*
dtype0*%
valueB"            *
_output_shapes
:
?
Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/minConst*
dtype0*,
_class"
 loc:@stream_5/depthwise_kernel*
valueB
 *b???*
_output_shapes
: 
?
Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/maxConst*
dtype0*,
_class"
 loc:@stream_5/depthwise_kernel*
valueB
 *b??>*
_output_shapes
: 
?
`stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*,
_class"
 loc:@stream_5/depthwise_kernel*
dtype0*
valueB"k?    *
_output_shapes
:
?
[stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter`stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*,
_class"
 loc:@stream_5/depthwise_kernel*
Tseed0* 
_output_shapes
::
?
[stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*,
_class"
 loc:@stream_5/depthwise_kernel*
dtype0*
value	B :*
_output_shapes
: 
?
Wstream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Dstream_5/depthwise_kernel/Initializer/stateless_random_uniform/shape[stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter]stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1[stream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
dtype0*
Tshape0*,
_class"
 loc:@stream_5/depthwise_kernel*&
_output_shapes
:
?
Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/subSubBstream_5/depthwise_kernel/Initializer/stateless_random_uniform/maxBstream_5/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_5/depthwise_kernel*
T0*
_output_shapes
: 
?
Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/mulMulWstream_5/depthwise_kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/sub*,
_class"
 loc:@stream_5/depthwise_kernel*
T0*&
_output_shapes
:
?
>stream_5/depthwise_kernel/Initializer/stateless_random_uniformAddV2Bstream_5/depthwise_kernel/Initializer/stateless_random_uniform/mulBstream_5/depthwise_kernel/Initializer/stateless_random_uniform/min*,
_class"
 loc:@stream_5/depthwise_kernel*
T0*&
_output_shapes
:
?
stream_5/depthwise_kernelVarHandleOp*
	container *
allowed_devices
 *,
_class"
 loc:@stream_5/depthwise_kernel*
_output_shapes
: *
dtype0*
shape:**
shared_namestream_5/depthwise_kernel
?
:stream_5/depthwise_kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_5/depthwise_kernel*
_output_shapes
: 
?
 stream_5/depthwise_kernel/AssignAssignVariableOpstream_5/depthwise_kernel>stream_5/depthwise_kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
?
-stream_5/depthwise_kernel/Read/ReadVariableOpReadVariableOpstream_5/depthwise_kernel*
dtype0*&
_output_shapes
:
?
stream_5/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@stream_5/bias*
valueB*    *
_output_shapes
:
?
stream_5/biasVarHandleOp*
	container *
allowed_devices
 * 
_class
loc:@stream_5/bias*
_output_shapes
: *
dtype0*
shape:*
shared_namestream_5/bias
k
.stream_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpstream_5/bias*
_output_shapes
: 
{
stream_5/bias/AssignAssignVariableOpstream_5/biasstream_5/bias/Initializer/zeros*
dtype0*
validate_shape( 
k
!stream_5/bias/Read/ReadVariableOpReadVariableOpstream_5/bias*
dtype0*
_output_shapes
:
?
4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpstream_5/depthwise_kernel*
dtype0*&
_output_shapes
:
?
+stream_5/depthwise_conv2d_3/depthwise/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:
?
3stream_5/depthwise_conv2d_3/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
%stream_5/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativeactivation_3/Relu4stream_5/depthwise_conv2d_3/depthwise/ReadVariableOp*
	dilations
*
explicit_paddings
 *
T0*&
_output_shapes
:*
strides
*
data_formatNHWC*
paddingVALID
|
2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpstream_5/bias*
dtype0*
_output_shapes
:
?
#stream_5/depthwise_conv2d_3/BiasAddBiasAdd%stream_5/depthwise_conv2d_3/depthwise2stream_5/depthwise_conv2d_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*&
_output_shapes
:
?
,batch_normalization_4/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0*
valueB*  ??*
_output_shapes
:
?
batch_normalization_4/gammaVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *.
_class$
" loc:@batch_normalization_4/gamma*
shape:*,
shared_namebatch_normalization_4/gamma
?
<batch_normalization_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
?
"batch_normalization_4/gamma/AssignAssignVariableOpbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*
dtype0*
validate_shape( 
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes
:
?
,batch_normalization_4/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_4/beta*
dtype0*
valueB*    *
_output_shapes
:
?
batch_normalization_4/betaVarHandleOp*
	container *
allowed_devices
 *-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta
?
;batch_normalization_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
?
!batch_normalization_4/beta/AssignAssignVariableOpbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*
dtype0*
validate_shape( 
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
dtype0*
_output_shapes
:
?
3batch_normalization_4/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
valueB*    *
_output_shapes
:
?
!batch_normalization_4/moving_meanVarHandleOp*
	container *
allowed_devices
 *4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean
?
Bbatch_normalization_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
?
(batch_normalization_4/moving_mean/AssignAssignVariableOp!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*
dtype0*
validate_shape( 
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
?
6batch_normalization_4/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
valueB*  ??*
_output_shapes
:
?
%batch_normalization_4/moving_varianceVarHandleOp*
	container *
allowed_devices
 *8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance
?
Fbatch_normalization_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
?
,batch_normalization_4/moving_variance/AssignAssignVariableOp%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*
dtype0*
validate_shape( 
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
?
batch_normalization_4/condIfkeras_learning_phasebatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#stream_5/depthwise_conv2d_3/BiasAdd*&
_read_only_resource_inputs
*7
else_branch(R&
$batch_normalization_4_cond_false_561*6
then_branch'R%
#batch_normalization_4_cond_true_560*
Tcond0
*@
_output_shapes.
,:::: : : : : : : *
_lower_using_switch_merge(*?
output_shapes.
,:::: : : : : : : *
Tout
2
*
Tin	
2
|
#batch_normalization_4/cond/IdentityIdentitybatch_normalization_4/cond*
T0*&
_output_shapes
:
t
%batch_normalization_4/cond/Identity_1Identitybatch_normalization_4/cond:1*
T0*
_output_shapes
:
t
%batch_normalization_4/cond/Identity_2Identitybatch_normalization_4/cond:2*
T0*
_output_shapes
:
p
%batch_normalization_4/cond/Identity_3Identitybatch_normalization_4/cond:3*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_4Identitybatch_normalization_4/cond:4*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_5Identitybatch_normalization_4/cond:5*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_6Identitybatch_normalization_4/cond:6*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_7Identitybatch_normalization_4/cond:7*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_8Identitybatch_normalization_4/cond:8*
T0*
_output_shapes
: 
p
%batch_normalization_4/cond/Identity_9Identitybatch_normalization_4/cond:9*
T0*
_output_shapes
: 
?
batch_normalization_4/cond_1StatelessIfkeras_learning_phase* 
_read_only_resource_inputs
 *9
else_branch*R(
&batch_normalization_4_cond_1_false_629*8
then_branch)R'
%batch_normalization_4_cond_1_true_628*
Tcond0
*
_output_shapes
: *
_lower_using_switch_merge(*
output_shapes
: *
Tout
2*	
Tin
 
p
%batch_normalization_4/cond_1/IdentityIdentitybatch_normalization_4/cond_1*
T0*
_output_shapes
: 
?
+batch_normalization_4/AssignMovingAvg/sub/xConst*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
)batch_normalization_4/AssignMovingAvg/subSub+batch_normalization_4/AssignMovingAvg/sub/x%batch_normalization_4/cond_1/Identity*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
?
+batch_normalization_4/AssignMovingAvg/sub_1Sub4batch_normalization_4/AssignMovingAvg/ReadVariableOp%batch_normalization_4/cond/Identity_1*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0*
_output_shapes
:
?
)batch_normalization_4/AssignMovingAvg/mulMul+batch_normalization_4/AssignMovingAvg/sub_1)batch_normalization_4/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0*
_output_shapes
:
?
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*
dtype0*&
 _has_manual_control_dependencies(*4
_class*
(&loc:@batch_normalization_4/moving_mean
?
6batch_normalization_4/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_4/moving_mean:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:
?
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*
dtype0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
valueB
 *  ??*
_output_shapes
: 
?
+batch_normalization_4/AssignMovingAvg_1/subSub-batch_normalization_4/AssignMovingAvg_1/sub/x%batch_normalization_4/cond_1/Identity*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0*
_output_shapes
: 
?
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
?
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp%batch_normalization_4/cond/Identity_2*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0*
_output_shapes
:
?
+batch_normalization_4/AssignMovingAvg_1/mulMul-batch_normalization_4/AssignMovingAvg_1/sub_1+batch_normalization_4/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0*
_output_shapes
:
?
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_4/moving_variance*&
 _has_manual_control_dependencies(*
dtype0
?
8batch_normalization_4/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_4/moving_variance<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:
o
activation_4/ReluRelu#batch_normalization_4/cond/Identity*
T0*&
_output_shapes
:
?
dropout/condIfkeras_learning_phaseactivation_4/Relu* 
_read_only_resource_inputs
 *)
else_branchR
dropout_cond_false_650*(
then_branchR
dropout_cond_true_649*
Tcond0
*4
_output_shapes"
 :: : : : : : : *
_lower_using_switch_merge(*3
output_shapes"
 :: : : : : : : *
Tout

2*
Tin
2
`
dropout/cond/IdentityIdentitydropout/cond*
T0*&
_output_shapes
:
T
dropout/cond/Identity_1Identitydropout/cond:1*
T0*
_output_shapes
: 
T
dropout/cond/Identity_2Identitydropout/cond:2*
T0*
_output_shapes
: 
T
dropout/cond/Identity_3Identitydropout/cond:3*
T0*
_output_shapes
: 
T
dropout/cond/Identity_4Identitydropout/cond:4*
T0*
_output_shapes
: 
T
dropout/cond/Identity_5Identitydropout/cond:5*
T0*
_output_shapes
: 
T
dropout/cond/Identity_6Identitydropout/cond:6*
T0*
_output_shapes
: 
T
dropout/cond/Identity_7Identitydropout/cond:7*
T0*
_output_shapes
: 
g
stream_6/flatten/ConstConst*
dtype0*
valueB"????0   *
_output_shapes
:
?
stream_6/flatten/ReshapeReshapedropout/cond/Identitystream_6/flatten/Const*
Tshape0*
T0*
_output_shapes

:0
?
7dense/kernel/Initializer/stateless_random_uniform/shapeConst*
_class
loc:@dense/kernel*
dtype0*
valueB"0   
   *
_output_shapes
:
?
5dense/kernel/Initializer/stateless_random_uniform/minConst*
_class
loc:@dense/kernel*
dtype0*
valueB
 *.???*
_output_shapes
: 
?
5dense/kernel/Initializer/stateless_random_uniform/maxConst*
dtype0*
_class
loc:@dense/kernel*
valueB
 *.??>*
_output_shapes
: 
?
Sdense/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
_class
loc:@dense/kernel*
dtype0*
valueB"?ß
    *
_output_shapes
:
?
Ndense/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterSdense/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*
_class
loc:@dense/kernel*
Tseed0* 
_output_shapes
::
?
Ndense/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_class
loc:@dense/kernel*
dtype0*
value	B :*
_output_shapes
: 
?
Jdense/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV27dense/kernel/Initializer/stateless_random_uniform/shapeNdense/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterPdense/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1Ndense/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*
_class
loc:@dense/kernel*
Tshape0*
dtype0*
_output_shapes

:0

?
5dense/kernel/Initializer/stateless_random_uniform/subSub5dense/kernel/Initializer/stateless_random_uniform/max5dense/kernel/Initializer/stateless_random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
: 
?
5dense/kernel/Initializer/stateless_random_uniform/mulMulJdense/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV25dense/kernel/Initializer/stateless_random_uniform/sub*
_class
loc:@dense/kernel*
T0*
_output_shapes

:0

?
1dense/kernel/Initializer/stateless_random_uniformAddV25dense/kernel/Initializer/stateless_random_uniform/mul5dense/kernel/Initializer/stateless_random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes

:0

?
dense/kernelVarHandleOp*
	container *
allowed_devices
 *
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
shape
:0
*
shared_namedense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
?
dense/kernel/AssignAssignVariableOpdense/kernel1dense/kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:0

?
dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
dtype0*
valueB
*    *
_output_shapes
:

?

dense/biasVarHandleOp*
	container *
allowed_devices
 *
_class
loc:@dense/bias*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
r
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0*
validate_shape( 
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:

h
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:0

?
dense/MatMulMatMulstream_6/flatten/Reshapedense/MatMul/ReadVariableOp*
transpose_b( *
T0*
transpose_a( *
_output_shapes

:

c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:

?
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*
_output_shapes

:

J

dense/ReluReludense/BiasAdd*
T0*
_output_shapes

:

?
9dense_1/kernel/Initializer/stateless_random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
dtype0*
valueB"
      *
_output_shapes
:
?
7dense_1/kernel/Initializer/stateless_random_uniform/minConst*!
_class
loc:@dense_1/kernel*
dtype0*
valueB
 *?5?*
_output_shapes
: 
?
7dense_1/kernel/Initializer/stateless_random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
dtype0*
valueB
 *?5?*
_output_shapes
: 
?
Udense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seedConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB"???0    *
_output_shapes
:
?
Pdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterUdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter/seed*!
_class
loc:@dense_1/kernel*
Tseed0* 
_output_shapes
::
?
Pdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/algConst*!
_class
loc:@dense_1/kernel*
dtype0*
value	B :*
_output_shapes
: 
?
Ldense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV29dense_1/kernel/Initializer/stateless_random_uniform/shapePdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounterRdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomGetKeyCounter:1Pdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV2/alg*!
_class
loc:@dense_1/kernel*
Tshape0*
dtype0*
_output_shapes

:

?
7dense_1/kernel/Initializer/stateless_random_uniform/subSub7dense_1/kernel/Initializer/stateless_random_uniform/max7dense_1/kernel/Initializer/stateless_random_uniform/min*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
: 
?
7dense_1/kernel/Initializer/stateless_random_uniform/mulMulLdense_1/kernel/Initializer/stateless_random_uniform/StatelessRandomUniformV27dense_1/kernel/Initializer/stateless_random_uniform/sub*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:

?
3dense_1/kernel/Initializer/stateless_random_uniformAddV27dense_1/kernel/Initializer/stateless_random_uniform/mul7dense_1/kernel/Initializer/stateless_random_uniform/min*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:

?
dense_1/kernelVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel*
shape
:
*
shared_namedense_1/kernel
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
?
dense_1/kernel/AssignAssignVariableOpdense_1/kernel3dense_1/kernel/Initializer/stateless_random_uniform*
dtype0*
validate_shape( 
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

?
dense_1/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_1/bias*
valueB*    *
_output_shapes
:
?
dense_1/biasVarHandleOp*
	container *
allowed_devices
 *
dtype0*
_output_shapes
: *
_class
loc:@dense_1/bias*
shape:*
shared_namedense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
x
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0*
validate_shape( 
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:

?
dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
transpose_b( *
T0*
transpose_a( *
_output_shapes

:
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
?
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*
_output_shapes

:
T
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*
T0*
_output_shapes

:ݪ
?
?
$batch_normalization_2_cond_false_3028
*readvariableop_batch_normalization_2_gamma:9
+readvariableop_1_batch_normalization_2_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:8
4fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.:9:::::*
is_training( *
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :9:,(
&
_output_shapes
:9
?
3
&batch_normalization_4_cond_1_false_629	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
2
%batch_normalization_1_cond_1_true_240	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
0
#batch_normalization_cond_1_true_111	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
$batch_normalization_4_cond_false_5618
*readvariableop_batch_normalization_4_gamma:9
+readvariableop_1_batch_normalization_4_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:8
4fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.::::::*
is_training( *
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
3
&batch_normalization_1_cond_1_false_241	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
2
%batch_normalization_3_cond_1_true_499	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
2
%batch_normalization_4_cond_1_true_628	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
#batch_normalization_4_cond_true_5608
*readvariableop_batch_normalization_4_gamma:9
+readvariableop_1_batch_normalization_4_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance:8
4fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_5_depthwise_conv2d_3_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.::::::*
is_training(*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
2
%batch_normalization_2_cond_1_true_369	
constJ
ConstConst*
dtype0*
valueB
 *???=*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
!batch_normalization_cond_false_446
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:*
&fusedbatchnormv3_stream_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?s
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
dtype0*
_output_shapes
:v
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV3&fusedbatchnormv3_stream_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
epsilon%??'7*B
_output_shapes0
.:{:::::*
is_training( *
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :{:,(
&
_output_shapes
:{
?
?
#batch_normalization_3_cond_true_4318
*readvariableop_batch_normalization_3_gamma:9
+readvariableop_1_batch_normalization_3_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:8
4fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.::::::*
is_training(*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
3
&batch_normalization_3_cond_1_false_500	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
$batch_normalization_3_cond_false_4328
*readvariableop_batch_normalization_3_gamma:9
+readvariableop_1_batch_normalization_3_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance:8
4fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_4_depthwise_conv2d_2_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
epsilon%??'7*B
_output_shapes0
.::::::*
is_training( *
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : ::,(
&
_output_shapes
:
?
1
$batch_normalization_cond_1_false_112	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
#batch_normalization_1_cond_true_1728
*readvariableop_batch_normalization_1_gamma:9
+readvariableop_1_batch_normalization_1_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance:6
2fusedbatchnormv3_stream_1_depthwise_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV32fusedbatchnormv3_stream_1_depthwise_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.:<:::::*
is_training(*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :<:,(
&
_output_shapes
:<
?
?
dropout_cond_false_650
identity_activation_4_relu
identity
optionalnone
optionalnone_1
optionalnone_2
optionalnone_3
optionalnone_4
optionalnone_5
optionalnone_6a
IdentityIdentityidentity_activation_4_relu*
T0*&
_output_shapes
:4
OptionalNoneOptionalNone*
_output_shapes
: 6
OptionalNone_1OptionalNone*
_output_shapes
: 6
OptionalNone_2OptionalNone*
_output_shapes
: 6
OptionalNone_3OptionalNone*
_output_shapes
: 6
OptionalNone_4OptionalNone*
_output_shapes
: 6
OptionalNone_5OptionalNone*
_output_shapes
: 6
OptionalNone_6OptionalNone*
_output_shapes
: "+
optionalnone_3OptionalNone_3:optional:0"+
optionalnone_2OptionalNone_2:optional:0"+
optionalnone_6OptionalNone_6:optional:0"'
optionalnoneOptionalNone:optional:0"+
optionalnone_4OptionalNone_4:optional:0"+
optionalnone_5OptionalNone_5:optional:0"
identityIdentity:output:0"+
optionalnone_1OptionalNone_1:optional:0*%
_input_shapes
::, (
&
_output_shapes
:
?
?
dropout_cond_true_649!
dropout_mul_activation_4_relu
dropout_selectv2
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?R
dropout/ConstConst*
dtype0*
valueB
 *   @*
_output_shapes
: z
dropout/MulMuldropout_mul_activation_4_reludropout/Const:output:0*
T0*&
_output_shapes
:f
dropout/ShapeConst*
dtype0*%
valueB"            *
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*

seed *
T0*
seed2 *&
_output_shapes
:[
dropout/GreaterEqual/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*&
_output_shapes
:T
dropout/Const_1Const*
dtype0*
valueB
 *    *
_output_shapes
: ?
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*&
_output_shapes
:?
OptionalFromValueOptionalFromValuedropout/Const:output:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValuedropout/Mul:z:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValuedropout/Shape:output:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue-dropout/random_uniform/RandomUniform:output:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValuedropout/GreaterEqual/y:output:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValuedropout/GreaterEqual:z:0*
Toutput_types
2
*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValuedropout/Const_1:output:0*
Toutput_types
2*
_output_shapes
: :???"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"-
dropout_selectv2dropout/SelectV2:output:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*%
_input_shapes
::, (
&
_output_shapes
:
?
?
 batch_normalization_cond_true_436
(readvariableop_batch_normalization_gamma:7
)readvariableop_1_batch_normalization_beta:M
?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean:S
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance:*
&fusedbatchnormv3_stream_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?s
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
dtype0*
_output_shapes
:v
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV3&fusedbatchnormv3_stream_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.:{:::::*
is_training(*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :{:,(
&
_output_shapes
:{
?
3
&batch_normalization_2_cond_1_false_370	
constJ
ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: "
constConst:output:0*
_input_shapes 
?
?
#batch_normalization_2_cond_true_3018
*readvariableop_batch_normalization_2_gamma:9
+readvariableop_1_batch_normalization_2_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance:8
4fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV34fusedbatchnormv3_stream_2_depthwise_conv2d_1_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.:9:::::*
is_training(*
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :9:,(
&
_output_shapes
:9
?
?
$batch_normalization_1_cond_false_1738
*readvariableop_batch_normalization_1_gamma:9
+readvariableop_1_batch_normalization_1_beta:O
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean:U
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance:6
2fusedbatchnormv3_stream_1_depthwise_conv2d_biasadd
fusedbatchnormv3
fusedbatchnormv3_0
fusedbatchnormv3_1
optionalfromvalue
optionalfromvalue_1
optionalfromvalue_2
optionalfromvalue_3
optionalfromvalue_4
optionalfromvalue_5
optionalfromvalue_6?u
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
dtype0*
_output_shapes
:x
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
dtype0*
_output_shapes
:?
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
dtype0*
_output_shapes
:?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
dtype0*
_output_shapes
:?
FusedBatchNormV3FusedBatchNormV32fusedbatchnormv3_stream_1_depthwise_conv2d_biasaddReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
T0*
epsilon%??'7*B
_output_shapes0
.:<:::::*
is_training( *
data_formatNHWC*
exponential_avg_factor%  ???
OptionalFromValueOptionalFromValueReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_1OptionalFromValueReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_2OptionalFromValue'FusedBatchNormV3/ReadVariableOp:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_3OptionalFromValue)FusedBatchNormV3/ReadVariableOp_1:value:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_4OptionalFromValue"FusedBatchNormV3:reserve_space_1:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_5OptionalFromValue"FusedBatchNormV3:reserve_space_2:0*
Toutput_types
2*
_output_shapes
: :????
OptionalFromValue_6OptionalFromValue"FusedBatchNormV3:reserve_space_3:0*
Toutput_types
2*
_output_shapes
: :???"1
optionalfromvalueOptionalFromValue:optional:0"5
optionalfromvalue_5OptionalFromValue_5:optional:0"5
optionalfromvalue_2OptionalFromValue_2:optional:0"(
fusedbatchnormv3FusedBatchNormV3:y:0"5
optionalfromvalue_4OptionalFromValue_4:optional:0"3
fusedbatchnormv3_0FusedBatchNormV3:batch_mean:0"5
optionalfromvalue_3OptionalFromValue_3:optional:0"7
fusedbatchnormv3_1!FusedBatchNormV3:batch_variance:0"5
optionalfromvalue_6OptionalFromValue_6:optional:0"5
optionalfromvalue_1OptionalFromValue_1:optional:0*-
_input_shapes
: : : : :<:,(
&
_output_shapes
:<"?
"?,
	variables?,?,
?
stream/kernel:0stream/kernel/Assign#stream/kernel/Read/ReadVariableOp:0(24stream/kernel/Initializer/stateless_random_uniform:08
k
stream/bias:0stream/bias/Assign!stream/bias/Read/ReadVariableOp:0(2stream/bias/Initializer/zeros:08
?
batch_normalization/gamma:0 batch_normalization/gamma/Assign/batch_normalization/gamma/Read/ReadVariableOp:0(2,batch_normalization/gamma/Initializer/ones:08
?
batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
?
!batch_normalization/moving_mean:0&batch_normalization/moving_mean/Assign5batch_normalization/moving_mean/Read/ReadVariableOp:0(23batch_normalization/moving_mean/Initializer/zeros:0@H
?
%batch_normalization/moving_variance:0*batch_normalization/moving_variance/Assign9batch_normalization/moving_variance/Read/ReadVariableOp:0(26batch_normalization/moving_variance/Initializer/ones:0@H
?
stream_1/depthwise_kernel:0 stream_1/depthwise_kernel/Assign/stream_1/depthwise_kernel/Read/ReadVariableOp:0(2@stream_1/depthwise_kernel/Initializer/stateless_random_uniform:08
s
stream_1/bias:0stream_1/bias/Assign#stream_1/bias/Read/ReadVariableOp:0(2!stream_1/bias/Initializer/zeros:08
?
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
?
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
?
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign7batch_normalization_1/moving_mean/Read/ReadVariableOp:0(25batch_normalization_1/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign;batch_normalization_1/moving_variance/Read/ReadVariableOp:0(28batch_normalization_1/moving_variance/Initializer/ones:0@H
?
stream_2/depthwise_kernel:0 stream_2/depthwise_kernel/Assign/stream_2/depthwise_kernel/Read/ReadVariableOp:0(2@stream_2/depthwise_kernel/Initializer/stateless_random_uniform:08
s
stream_2/bias:0stream_2/bias/Assign#stream_2/bias/Read/ReadVariableOp:0(2!stream_2/bias/Initializer/zeros:08
?
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
?
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
?
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign7batch_normalization_2/moving_mean/Read/ReadVariableOp:0(25batch_normalization_2/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign;batch_normalization_2/moving_variance/Read/ReadVariableOp:0(28batch_normalization_2/moving_variance/Initializer/ones:0@H
?
stream_4/depthwise_kernel:0 stream_4/depthwise_kernel/Assign/stream_4/depthwise_kernel/Read/ReadVariableOp:0(2@stream_4/depthwise_kernel/Initializer/stateless_random_uniform:08
s
stream_4/bias:0stream_4/bias/Assign#stream_4/bias/Read/ReadVariableOp:0(2!stream_4/bias/Initializer/zeros:08
?
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2.batch_normalization_3/gamma/Initializer/ones:08
?
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2.batch_normalization_3/beta/Initializer/zeros:08
?
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign7batch_normalization_3/moving_mean/Read/ReadVariableOp:0(25batch_normalization_3/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign;batch_normalization_3/moving_variance/Read/ReadVariableOp:0(28batch_normalization_3/moving_variance/Initializer/ones:0@H
?
stream_5/depthwise_kernel:0 stream_5/depthwise_kernel/Assign/stream_5/depthwise_kernel/Read/ReadVariableOp:0(2@stream_5/depthwise_kernel/Initializer/stateless_random_uniform:08
s
stream_5/bias:0stream_5/bias/Assign#stream_5/bias/Read/ReadVariableOp:0(2!stream_5/bias/Initializer/zeros:08
?
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2.batch_normalization_4/gamma/Initializer/ones:08
?
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2.batch_normalization_4/beta/Initializer/zeros:08
?
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign7batch_normalization_4/moving_mean/Read/ReadVariableOp:0(25batch_normalization_4/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign;batch_normalization_4/moving_variance/Read/ReadVariableOp:0(28batch_normalization_4/moving_variance/Initializer/ones:0@H
?
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(23dense/kernel/Initializer/stateless_random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(25dense_1/kernel/Initializer/stateless_random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"?
trainable_variables??
?
stream/kernel:0stream/kernel/Assign#stream/kernel/Read/ReadVariableOp:0(24stream/kernel/Initializer/stateless_random_uniform:08
k
stream/bias:0stream/bias/Assign!stream/bias/Read/ReadVariableOp:0(2stream/bias/Initializer/zeros:08
?
batch_normalization/gamma:0 batch_normalization/gamma/Assign/batch_normalization/gamma/Read/ReadVariableOp:0(2,batch_normalization/gamma/Initializer/ones:08
?
batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
?
stream_1/depthwise_kernel:0 stream_1/depthwise_kernel/Assign/stream_1/depthwise_kernel/Read/ReadVariableOp:0(2@stream_1/depthwise_kernel/Initializer/stateless_random_uniform:08
s
stream_1/bias:0stream_1/bias/Assign#stream_1/bias/Read/ReadVariableOp:0(2!stream_1/bias/Initializer/zeros:08
?
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
?
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
?
stream_2/depthwise_kernel:0 stream_2/depthwise_kernel/Assign/stream_2/depthwise_kernel/Read/ReadVariableOp:0(2@stream_2/depthwise_kernel/Initializer/stateless_random_uniform:08
s
stream_2/bias:0stream_2/bias/Assign#stream_2/bias/Read/ReadVariableOp:0(2!stream_2/bias/Initializer/zeros:08
?
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
?
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
?
stream_4/depthwise_kernel:0 stream_4/depthwise_kernel/Assign/stream_4/depthwise_kernel/Read/ReadVariableOp:0(2@stream_4/depthwise_kernel/Initializer/stateless_random_uniform:08
s
stream_4/bias:0stream_4/bias/Assign#stream_4/bias/Read/ReadVariableOp:0(2!stream_4/bias/Initializer/zeros:08
?
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2.batch_normalization_3/gamma/Initializer/ones:08
?
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2.batch_normalization_3/beta/Initializer/zeros:08
?
stream_5/depthwise_kernel:0 stream_5/depthwise_kernel/Assign/stream_5/depthwise_kernel/Read/ReadVariableOp:0(2@stream_5/depthwise_kernel/Initializer/stateless_random_uniform:08
s
stream_5/bias:0stream_5/bias/Assign#stream_5/bias/Read/ReadVariableOp:0(2!stream_5/bias/Initializer/zeros:08
?
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2.batch_normalization_4/gamma/Initializer/ones:08
?
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2.batch_normalization_4/beta/Initializer/zeros:08
?
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(23dense/kernel/Initializer/stateless_random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
?
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(25dense_1/kernel/Initializer/stateless_random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08p?6