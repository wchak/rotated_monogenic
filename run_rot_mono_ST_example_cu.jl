using CUDA
using ScatteringTransform
using MonogenicFilterFlux


# need to add dependency
using FileIO, JLD2

#st = stFlux((10,10,1,3), 2);
#output = st(rand(10,10,1,3));

#st = stFlux((10,1,3), 2);
#st = cu(st);
#output = st(cu(rand(10, 10, 1, 3)));

using MLDatasets;

train_x, train_y = MNIST.traindata();
test_x,  test_y  = MNIST.testdata();

nTrain_x, nTrain_y, nTrain = size(train_x);
nTest_x, nTest_y, nTest = size(test_x);

train_x = reshape(train_x, (nTrain_x, nTrain_y, 1, nTrain));
train_x = Float32.(train_x);

test_x = reshape(test_x, (nTest_x, nTest_y, 1, nTest));
test_x = Float32.(test_x);

nGroupSubsample = 50;
nSubsample = nGroupSubsample * 10;

ang = pi / 4;

# train
for set_ = 0:119
	print("train")
	print(set_)
	print("\n")

	scale = 4;

	st = stFlux((nTrain_x, nTrain_y, 1, nSubsample), 2, σ=abs, outputPool = 1, scale = scale);
	st = cu(st);

	input_data = train_x[:,:,1:1,(nSubsample * set_ + 1):(nSubsample * set_ + nSubsample)];
	output = st(cu(input_data));

	# rotated monogenic
	img_rotated = rotate_image(input_data, ang);
	nTrain_x_rot, nTrain_y_rot = size(img_rotated);
	st = stFlux((nTrain_x_rot, nTrain_y_rot, 1, nSubsample), 2, σ=abs, outputPool = 1, scale = scale);
	st = cu(st);
	output_rot = st(cu(img_rotated));

	output_1 = Array(output[1]);
	output_2 = Array(output[2]);

	output_rot_1 = Array(output_rot[1]);
	output_rot_2 = Array(output_rot[2]);

	output_rot_1 = inv_rotate_out(output_rot_1, img_rotated, -ang);
	output_rot_2 = inv_rotate_out(output_rot_2, img_rotated, -ang);

	output_rot_1 = cat(output_1, output_rot_1, dims = 4);
	output_rot_2 = cat(output_2, output_rot_2, dims = 5);


	FileIO.save("output/output_mnist_2_1_mat_scale_" * string(scale) * "_set_" * string(set_) * ".jld2","output",output_rot_1);
	FileIO.save("output/output_mnist_2_2_mat_scale_" * string(scale) * "_set_" * string(set_) * ".jld2","output",output_rot_2);

end

# 19
for set_ = 0:19

	print(set_)
	print("\n")
	global test_y;
	global test_x;

	scale = 4;

	st = stFlux((nTest_x, nTest_y, 1, nSubsample), 2, σ=abs, outputPool = 1, scale = scale);
	st = cu(st);

	input_data = test_x[:,:,1:1,(nSubsample * set_ + 1):(nSubsample * set_ + nSubsample)];
	output = st(cu(input_data));

	# rotated monogenic
	img_rotated = rotate_image(input_data, ang);
	nTest_x_rot, nTest_t_rot = size(img_rotated);
	st = stFlux((nTest_x_rot, nTest_y_rot, 1, nSubsample), 2, σ=abs, outputPool = 1, scale = scale);
	st = cu(st);
	output_rot = st(cu(img_rotated));

	output_1 = Array(output[1]);
	output_2 = Array(output[2]);

	output_rot_1 = Array(output_rot[1]);
	output_rot_2 = Array(output_rot[2]);

	output_rot_1 = inv_rotate_out(output_rot_1, img_rotated, -ang);
	output_rot_2 = inv_rotate_out(output_rot_2, img_rotated, -ang);

	output_rot_1 = cat(output_1, output_rot_1, dims = 4);
	output_rot_2 = cat(output_2, output_rot_2, dims = 5);


	FileIO.save("output/output_mnist_2_1_mat_scale_" * string(scale) * "_set_" * string(set_) * "_test.jld2","output",output_rot_1);

	FileIO.save("output/output_mnist_2_2_mat_scale_" * string(scale) * "_set_" * string(set_) * "_test.jld2","output",output_rot_2);

end
