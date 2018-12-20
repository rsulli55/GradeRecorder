

module NNet
#using MNIST
using  Images, ImageView, Random


mutable struct NeuralNet

    num_layers::Int64
    sizes::Array{Int64, 1}
    biases
    weights

# the weights should be normal with mean 0 and variance 1/sqrt(n)
    NeuralNet(sizes::Array{Int64, 1}) =
        new(length(sizes), sizes, [randn(y, 1) for y in sizes[2:end]],
                [randn(y, x)/sqrt(x) for (x,y) in zip(sizes[1:end-1], sizes[2:end])])

    #= possibly add a constructor here =#
end



function sigmoid(z)
    return 1.0 ./ (1.0 .+ exp.(-z))
end
function sigmoid_prime(z)
    return sigmoid(z) .* (1 .- sigmoid(z))
end

function cost_derivative(output_acts, y)
    # println("acts: $output_acts")
    # println("y: $y")
    # println(output_acts - y)
    return (output_acts - y)
end

function feedforward(net::NeuralNet, a::Array{Float64,2})
    act = a
    for (b, w) in zip(net.biases, net.weights)
        act = sigmoid(w * act + b)
    end

    return act
end


function SGD(net::NeuralNet, training_data, epochs::Int64, mini_batch_size::Int64,eta::Float64, lambda::Float64, test_data=[])

    if !isempty(test_data)
        n_test = length(test_data)
    end

    n = length(training_data)
    # println(n)

    for j in 1:epochs
        shuffle!(training_data)

        # println("shuffled")

        mini_batches = [training_data[k:k+mini_batch_size-1] for k in 1:mini_batch_size:n]

        # println("got mini_batch")
        for mb in mini_batches
            #println("mb=$mb")
            update_mini_batch(net, mb, eta, lambda, n)
        end

        if !isempty(test_data)
            n_correct = evaluate(net, test_data)
            println("Epoch $j: $n_correct / $n_test")
        else
            println("Epoch $j complete.")
        end
    end
end

function SGD(net::NeuralNet, trainfiles::Int64, epochs::Int64, mini_batch_size::Int64,eta::Float64, lambda::Float64)

    # if !isempty(test_data)
    #     n_test = length(test_data)
    # end


    # println(n)

    for j in 1:epochs
        for t in 1:trainfiles
            training_data, validation_data = load_data(t)

            n = length(training_data)
            # shuffle!(training_data)

            # println("shuffled")

            mini_batches = [training_data[k:k+mini_batch_size-1] for k in 1:mini_batch_size:n]

            # println("got mini_batch")
            for mb in mini_batches
                #println("mb=$mb")
                update_mini_batch(net, mb, eta, lambda, n*trainfiles)
                ## the total amount of trainding data is n * trainfiles
            end

            #print how well it is doing on validation data
            # if !isempty(test_data)
            n_correct = evaluate(net, validation_data)
            println("Epoch $j: $n_correct / $(length(validation_data))")
            # else
            # println("Epoch $j complete.")
        end
    end
end

function update_mini_batch(net::NeuralNet, mini_batch, eta::Float64, lambda::Float64, n::Int64)
    nabla_b = [zeros(size(b)) for b in net.biases]
    nabla_w = [zeros(size(w)) for w in net.weights]


    for (x,y) in mini_batch
        #println("x = $x, y=$y")

        (delta_nabla_b, delta_nabla_w) = backprop(net, x, y)
        # println("nabla_b")
        # println(nabla_b)

        nabla_b = [nb + dnb for (nb, dnb) in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for (nw, dnw) in zip(nabla_w, delta_nabla_w)]
    end

    net.weights = [(1 - eta*(lambda/n)) * w - (eta/length(mini_batch)) * nw for (w, nw) in zip(net.weights, nabla_w)]
    net.biases = [b - (eta/length(mini_batch)) * nb for (b, nb) in zip(net.biases, nabla_b)]
end

function backprop(net::NeuralNet, x, y)
    nabla_b = [zeros(size(b)) for b in net.biases]
    nabla_w = [zeros(size(w)) for w in net.weights]

    # starting activation is x
    act = x
    # println(typeof(x))
    #keep track of the activations
    acts = Vector{Array{Float64, 2}}(undef, net.num_layers)
    acts[1] = x
    zs = Vector{Array{Float64, 2}}(undef, net.num_layers - 1)
    i = 1

    for (b,w) in zip(net.biases, net.weights)
        # println("act = $(size(act)), w = $(size(w))")
        z = w * act + b
        # println("z = $(size(z))")
        zs[i] = z
        act = sigmoid(z)
        # println("act = $(size(act))")
        i += 1
        acts[i] = act

    end
    # println("Out of for loop")

    #println("acts: $(size(acts[end])), y: $(size(y))")
    ## QUADRATIC COST DELTA
    ## delta = cost_derivative(acts[end], y) .* sigmoid_prime(zs[end])

    ### CROSS_ENTROPY COST DELTA
    delta = acts[end] - y
    # println("delta")

    nabla_b[end] = delta
    # println("delta = $(size(delta)), trans(act[end-1]) = $(size(transpose(acts[end-1]))) ")
    nabla_w[end] = delta * transpose(acts[end-1])
    #
    # println("zs: $(size(zs)), acts: $(size(acts)), nabla_w: $(size(nabla_w))")
    # println("num_layers = $(net.num_layers)")
    #
    for l in 1:net.num_layers - 2
        z = zs[end - l]
        sp = sigmoid_prime(z)
        # println("$(size(sp))")

        delta = (transpose(net.weights[end-l+1]) * delta)  .* sp
        # println("delta = $(size(delta))")
        nabla_b[end-l] = delta
        nabla_w[end-l] = delta * transpose(acts[end-l-1])
    end

    # println("nabla_b")
    # println(nabla_b)
    # println("nabla_w")
    # println(nabla_w)
    return (nabla_b, nabla_w)
end

function vectorize_label(l::Float64)
    res = zeros(10, 1)
    res[Int(l)] = 1

    return res
end

function load_data(n::Int64)
    # test_data = [(reshape(testfeatures(i), (784,1))/255.0, vectorize_label(testlabel(i)+1)) for i in 1:10_000]

    if n == 0
        train_data = [(reshape(trainfeatures(i), (784,1))/255.0, vectorize_label(trainlabel(i)+1)) for i in 1:60_000]

    else
        #Note changed path
    A = load("/home/ryan/Julia/transformedMNIST$n/images$n.png")

    B = Array{Float64, 1}(undef, size(A)[2])
    read!("/home/ryan/Julia/transformedMNIST$n/labels$n.dat", B)
    # B = read(open("/home/ryan/Julia/transformedMNIST$n/labels$n.dat"), Float64, size(A)[2])
    train_data = Vector{Tuple{Array{Float64,2}, Array{Float64, 2}}}(undef, size(A)[2])
    #println("After traindata")


    for i in 1:size(A)[2]
        #println("i = $i")
        dig = A[:, i]
        newDig = Array{Float64, 2}(undef, 784, 1)
        for j in eachindex(dig)
            #println("j = $j")
            newDig[j] = convert(Int,dig[j].val.i) / 255.0
        end
        train_data[i] = (newDig, vectorize_label(B[i]+1))
    end
end
    perm = randperm(length(train_data))

    train_data = train_data[perm]
    validation_data = train_data[1:10_000]

    return (train_data[10_001:end], validation_data) #, test_data)
    # return train_data
end

function init_net(net::NeuralNet)
    net.biases = [randn(y, 1) for y in net.sizes[2:end]]
    net.weights = [randn(y, x) for (x,y) in zip(net.sizes[1:end-1], net.sizes[2:end])]
end

function save_net(net::NeuralNet)
    for i in 1:length(net.biases)
        f = open("/home/ryan/Julia/longtrainbiases$i.dat", "w")
        write(f, net.biases[i])
        close(f)
    end

    for i in 1:length(net.weights)
        f = open("/home/ryan/Julia/longtrainweights$i.dat", "w")
        write(f, net.weights[i])
        close(f)
    end

    f = open("/home/ryan/Julia/longtrainnetworknotes.txt", "w")
    write(f, "Network biases lengths\n")
    write(f, "biases[1] length : $(size(net.biases[1]))\n")
    write(f, "biases[2] length : $(size(net.biases[2]))\n")
    write(f, "Network weights lengths\n")
    write(f, "weights[1] length : $(size(net.weights[1]))\n")
    write(f, "weights[2] length : $(size(net.weights[2]))\n")
    write(f, "end of networknotes")
    close(f)
end

## eventually add directory
function load_net()
    # biases = Vector{typeof(net.biases[1])}
    # weights = Vector{typeof(net.weights[1])}
    ## maybe try and do this a better way

    net = NNet.NeuralNet([784,100,10])

    biases = Vector{Any}(nothing, length(net.biases))
    weights = Vector{Any}(nothing, length(net.weights))

    for i in 1:length(net.weights)
        B = similar(net.biases[i])
        W = similar(net.weights[i])
        read!("/home/ryan/Julia/longtrainbiases$i.dat", B)
        read!("/home/ryan/Julia/longtrainweights$i.dat", W)
        biases[i] = B
        weights[i] = W
    end

    net.biases = biases
    net.weights = weights

    return net
end

function evaluate(net, test_data)
    # predictions = [feedforward(net, x) for (x, _) in test_data]
    results = [vectorize_label(Float64(argmax(feedforward(net, x))[1])) for (x,y) in test_data]
    n_correct = 0

    for i in 1:length(test_data)
        if results[i] == test_data[i][2]
            n_correct += 1
        end
    end

    return n_correct
end
end
