using CSV, DataFrames
using Images, ImageView

push!(LOAD_PATH, "/home/ryan/Julia")
import NNet



function loadIDS(path::String, delim::Char)
    students = CSV.read(path, delim=delim)
    # students = CSV.read(path, delim=delim)
    intIds = students[:, 1]
    return intIds
end

function intToArr(n::Int64)
    arr = Int64[]
    while n > 0
        push!(arr, n % 10)
        n = floor(n /10)
    end

    return reverse(arr)
end

function arrToInt(arr)
    return parse(Int, join(arr))
end

function processIDS(directory::String, quizName::String, numIDS::Int64)
for l in 1:numIDS
    path = directory * quizName * "ID-$l.png"
    id = load(path)
    origBW = Gray.(id) .> 0.9
    # imshow(id)
    # blurred  = imfilter(id, Kernel.gaussian(0.15))

    blurred  = imfilter(id, Kernel.gaussian(0.35))
    # imshow(blurred)
    #gray = Gray.(blurred).> .90
    gray = Gray.(blurred) #.> 0.90
    bw = gray .> 0.9

    #gray = erode(gray)
    #imshow(gray)
    # for i in 3:10
    #     corners = fastcorners(gray, i)
    #     imshow(corners, name="number $i")
    # end

    #corners = fastcorners(gray, 4)

    labels = label_components(bw)
    boxes = component_boxes(labels)

    numBoxes = length(boxes)
    keep = fill(true, numBoxes)

    #h, w = size(gray)

    #remove any "big" boxes or any "small" boxes
    for i in 1:numBoxes
    b = boxes[i]
    # if abs(b[1][1] - 1) + abs(b[1][2] - 1) < 10 &&
    #     abs(b[2][1] - h) + abs(b[2][2] - w) < 10
    #     # println("In if")
    #     keep[i] = false
    # end

    if (b[2][1] - b[1][1]) * (b[2][2] - b[1][2]) <= 200 ||
    (b[2][1] - b[1][1]) * (b[2][2] - b[1][2]) >= 3600
    # println("We remove box $i because of its size")
        keep[i] = false
    end

    if (b[2][1] - b[1][1]) < 10 || (b[2][2] - b[1][2]) < 10
        keep[i] = false
    end

    end

    # println(length(keep), length(boxes))
    boxes1 = boxes[keep]
    numBoxes=length(boxes1)
    keep1 = fill(true, numBoxes)


    # for i in 3:10
    #remove overlapping boxes
    for i in 1:numBoxes, j in 1:numBoxes
        if j != i && keep1[i] && keep1[j] && compareBoxes(boxes1[i], boxes1[j])
    # cleprintln("box $j is in box $i")
        keep1[j] = false
        end
    end

    boxes2 = boxes1[keep1]
    #id_digits = Array{Array{Gray{Float64},2}, 1}(length(boxes2))
    #

    try
        mkdir(directory * quizName *"ID$l-digits/")
    catch E
        #println("directory already exists")
    end
    for k in eachindex(boxes2)

        b = boxes2[k]
        # println(b)
        # imshow(gray[b[1][1]:b[2][1], b[1][2]:b[2][2]])
        # println("The type of the digit is:", typeof(gray[b[1][1]:b[2][1], b[1][2]:b[2][2]]))
        dig_orig = origBW[b[1][1]+2:b[2][1]-2, b[1][2]+2:b[2][2]-2]
        dig_processed = bw[b[1][1]+2:b[2][1]-2, b[1][2]+2:b[2][2]-2]
        #println(size(dig))

        #try resizing digits later

        # sz = (28,28)
        # σ = map((o,n)->0.5*o/n, size(dig), sz)
        # kern = KernelFactors.gaussian(σ)   # from ImageFiltering
        # dig_resized = imresize(imfilter(dig, kern, NA()), sz)
        #id_digits[k] = dig_resized
        #
        # invert the colors of the digit
        # for i = 1:length(dig_resized)
        #     dig_resized[i] = xor(true, dig_resized[i]
        # end

        # imshow(dig_resized)
        ###
        ## change this back to dig_resized possibly!!!!!
        ####
        save(directory * quizName *"ID$l-digits/dig$(k)_orig.png", dig_orig)
        save(directory * quizName *"ID$l-digits/dig$(k)_processed.png", dig_processed)

    end
    # imshow(id)
#
    if length(boxes2) != 10
        println("ID $l has $(length(boxes2)) digits")
     #println("ID $l does not have 8 digits")
    end
end
end


    """directory is the path to the assignments directory"""
function saveGrades(directory::String, quizName::String, idFile::String, idDelim::Char,
    numIDS::Int64, numDigits::Int64)

    intIds = loadIDS(directory * idFile, idDelim)
    ids = [intToArr(id) for id in skipmissing(intIds)]
    println("ids")
    println(ids)

    predicted, _, _ = getPredictions(directory, quizName, numIDS, numDigits)
    predIDS = [p[1:8] for p in predicted]
    predScores = [arrToInt(p[9:10]) for p in predicted]
    println("predIDS")
    println(predIDS)

    idMap, indices = matchIDS(predIDS, ids, numIDS)
    println("map")
    println(idMap)
    println("indices")
    println(indices)

    ### keys has no order!!! may need to do it another way
    foundIDS = [intIds[idMap[i]] for i in 1:length(predicted) if indices[i]]
    println(foundIDS)

    grades = DataFrame(ID=foundIDS, Score=predScores)

    CSV.write(directory*quizName*"grades.csv", grades)
end


function matchIDS(predicted, known, numIDS)
    predsLeft = Set(1:length(predicted))
    knownLeft = Set(1:length(known))
    predToKnown = zeros(Int64, length(predicted))
    indices = zeros(Bool, length(predicted))

    numToMatch = 8

    while !isempty(predsLeft) && numToMatch >= 2
        for p in predsLeft
            for k in knownLeft
                if  !ismissing(k) && sum(predicted[p] .== known[k]) >= numToMatch
                    predToKnown[p] = k
                    indices[p] = true
                    delete!(predsLeft, p)
                    delete!(knownLeft, k)
                end
            end
        end
        numToMatch -= 1
    end

    if (!isempty(predsLeft))
        println("Could not match everything!")
        println(predsLeft)
    end

    return predToKnown, indices
end

function compareBoxes(b,c)
    #check if box c is contained in box b
    almostContained = c[1][1] >= b[1][1] -20 && c[1][2] >= b[1][2] -26 && c[2][1] <= b[2][1] + 20 && c[2][2] <= b[2][2] + 26
    smallerArea = (c[2][1] - c[1][1]) * (c[2][2] - c[1][2]) <= (b[2][1] - b[1][1]) * (b[2][2] - b[1][2])
    return almostContained && smallerArea
end

#l = 39
#### OLD PROCESSIDS
#####
# function processIDS(directory::String, numIDS::Int64)
# for l in 1:numIDS
#     path = directory * "density200crop-$l.png"
#     id = load(path)
#     origBW = Gray.(id) .> 0.9
#     # imshow(id)
#     # blurred  = imfilter(id, Kernel.gaussian(0.15))
#
#     blurred  = imfilter(id, Kernel.gaussian(0.35))
#     # imshow(blurred)
#     #gray = Gray.(blurred).> .90
#     gray = Gray.(blurred) #.> 0.90
#     bw = gray .> 0.9
#     #gray = erode(gray)
#     # imshow(gray)
#     # for i in 3:10
#     #     corners = fastcorners(gray, i)
#     #     imshow(corners, name="number $i")
#     # end
#
#     #corners = fastcorners(gray, 4)
#
#     labels = label_components(bw)
#     boxes = component_boxes(labels)
#
#     numBoxes = length(boxes)
#     keep = fill(true, numBoxes)
#
#     #h, w = size(gray)
#
#     #remove any "big" boxes or any "small" boxes
#     for i in 1:numBoxes
#     b = boxes[i]
#     # if abs(b[1][1] - 1) + abs(b[1][2] - 1) < 10 &&
#     #     abs(b[2][1] - h) + abs(b[2][2] - w) < 10
#     #     # println("In if")
#     #     keep[i] = false
#     # end
#
#     if (b[2][1] - b[1][1]) * (b[2][2] - b[1][2]) <= 200 ||
#     (b[2][1] - b[1][1]) * (b[2][2] - b[1][2]) >= 3600
#     # println("We remove box $i because of its size")
#         keep[i] = false
#     end
#
#     if (b[2][1] - b[1][1]) < 10 || (b[2][2] - b[1][2]) < 10
#         keep[i] = false
#     end
#
#     end
#
#     # println(length(keep), length(boxes))
#     boxes1 = boxes[keep]
#     numBoxes=length(boxes1)
#     keep1 = fill(true, numBoxes)
#
#
#     # for i in 3:10
#     #remove overlapping boxes
#     for i in 1:numBoxes, j in 1:numBoxes
#         if j != i && keep1[i] && keep1[j] && compareBoxes(boxes1[i], boxes1[j])
#     # cleprintln("box $j is in box $i")
#         keep1[j] = false
#         end
#     end
#
#     boxes2 = boxes1[keep1]
#     #id_digits = Array{Array{Gray{Float64},2}, 1}(length(boxes2))
#     #
#
#     try
#         mkdir(directory * "quiz$l-digits/")
#     catch E
#         #println("directory already exists")
#     end
#     for k in eachindex(boxes2)
#
#         b = boxes2[k]
#         # println(b)
#         # imshow(gray[b[1][1]:b[2][1], b[1][2]:b[2][2]])
#         # println("The type of the digit is:", typeof(gray[b[1][1]:b[2][1], b[1][2]:b[2][2]]))
#         dig_orig = origBW[b[1][1]+2:b[2][1]-2, b[1][2]+2:b[2][2]-2]
#         dig_processed = bw[b[1][1]+2:b[2][1]-2, b[1][2]+2:b[2][2]-2]
#         #println(size(dig))
#
#         #try resizing digits later
#
#         # sz = (28,28)
#         # σ = map((o,n)->0.5*o/n, size(dig), sz)
#         # kern = KernelFactors.gaussian(σ)   # from ImageFiltering
#         # dig_resized = imresize(imfilter(dig, kern, NA()), sz)
#         #id_digits[k] = dig_resized
#         #
#         # invert the colors of the digit
#         # for i = 1:length(dig_resized)
#         #     dig_resized[i] = xor(true, dig_resized[i]
#         # end
#
#         # imshow(dig_resized)
#         ###
#         ## change this back to dig_resized possibly!!!!!
#         ####
#         save(directory * "quiz$l-digits/dig$(k)_orig.png", dig_orig)
#         save(directory * "quiz$l-digits/dig$(k)_processed.png", dig_processed)
#
#     end
#     # imshow(id)
# #
#     if length(boxes2) != 10
#         println("ID $l has $(length(boxes2)) digits")
#      #println("ID $l does not have 8 digits")
#     end
# end
# end

function getPredictions(directory::String, quizName::String, numIDS::Int64, numDigits::Int64)
    ids = [[load(directory* quizName * "ID$l-digits/dig$(k)_orig.png") for k in 1:numDigits] for l in 1:numIDS]
    invertedIDS = []
    finalIDS =[]
    results = Vector{Vector{Int64}}(undef, numIDS)
    for l in 1:numIDS
        res = zeros(Int64, numDigits)
        invertedID = []
        finalID = []
        for k in 1:numDigits
            dig = ids[l][k]
            # println(size(dig))
            # imshow(dig)
            # invertedDig = zeros(length(dig), 1)
            invertedDig = zeros(size(dig))


            for i in 1:length(dig)
                #swap black and white and then normalize
                invertedDig[i] = (255 - convert(Int,dig[i].val.i)) / 255.0
            end
            #try to remove any fully black rows or columns
            ## the 1s are being classified as 8s and I think its because
            ## we are removing too much and then we are stretching it a lot
            ## when we resize
            ## so lets try keeping a counter so we do not remove too many rows/columns
            push!(invertedID, invertedDig)
            save(directory* quizName * "ID$l-digits/dig$(k)_invert.png", invertedDig)
            maxToRemove = 20
            if sum(invertedDig) < 200
                # println("setting max low")
                # println("ID number: $l, digit: $k")
                maxToRemove = 6
            end


            nRemoved = 0
            while sum(invertedDig[1, :]) == 0  && nRemoved < maxToRemove
                invertedDig = invertedDig[2:end, :]
                nRemoved += 1
            end
            nRemoved = 0
            while sum(invertedDig[end, :]) == 0  && nRemoved < maxToRemove
                invertedDig = invertedDig[1:end-1, :]
                nRemoved += 1
            end
            nRemoved = 0
            while sum(invertedDig[:, 1]) == 0  && nRemoved < maxToRemove
                invertedDig = invertedDig[:, 2:end]
                nRemoved += 1
            end
            nRemoved = 0
            while sum(invertedDig[:, end]) == 0  && nRemoved < maxToRemove
                invertedDig = invertedDig[:, 1:end-1]
                nRemoved += 1
            end

            sz = (20,20)
            σ = map((o,n)->0.5*o/n, size(invertedDig), sz)
            kern = KernelFactors.gaussian(σ)   # from ImageFiltering
            resizedDig = imresize(imfilter(invertedDig, kern, NA()), sz)


            ## find the center of mass
            rows,cols = size(resizedDig)
            #
            # totalMass = sum(resizedDig)
            # rowTotal = 0
            # for i in 1:rows
            #     rowTotal += dot(resizedDig[i, :], 1:cols)
            # end
            # centralRow = convert(Int,round(rowTotal / totalMass))
            # centralRow = centralRow < 1 ? 1 : centralRow
            # centralRow = centralRow > rows ? rows  : centralRow
            #
            # colTotal = 0
            # for j in 1:cols
            #     colTotal += dot(resizedDig[:, j], 1:rows)
            # end
            # centralCol = convert(Int, round(colTotal / totalMass))
            # centralCol = centralCol < 1 ? 1 : centralCol
            # centralCol = centralCol > cols ? cols : centralCol

            ## use middle of the image
            midRow = convert(Int, round(rows/2))
            midCol = convert(Int, round(cols/2))




            ## copy digit over to 28x28 matching up the center with the center of mass
            finalDig = zeros(28,28)
            for i in 1:rows, j in 1:cols
                finalDig[14 - midRow + i, 14 - midCol + j] = resizedDig[i, j]
            end
            # try
            #     for i in 1:rows, j in 1:cols
            #     # println("l = $l, k = $k")
            #     # println("i: $i, j: $j")
            #     # println("cRow: $centralRow, cCol: $centralCol")
            #         finalDig[14 - centralRow + i, 14 - centralCol + j] = resizedDig[i, j]
            #     end
            # catch E
            #     # println("In Catch block")
            #     # println("ID is $l, digit is $k")
            #
            #     for i in 1:rows, j in 1:cols
            #     # println("l = $l, k = $k")
            #     # println("i: $i, j: $j")
            #     # println("cRow: $centralRow, cCol: $centralCol")
            #         finalDig[14 - midRow + i, 14 - midCol + j] = resizedDig[i, j]
            #     end
            # end


            res[k] = argmax(NNet.feedforward(net, reshape(finalDig, (784,1))))[1] - 1

            push!(finalID, finalDig)
            save(directory* quizName * "ID$l-digits/dig$(k)_final.png", finalDig)
        end
        push!(invertedIDS, invertedID)
        push!(finalIDS, finalID)

        results[l] = res
    end

    return (results, invertedIDS, finalIDS)
end


print("Enter the (full path to) quizzes directory: ")
quizDirectory = readline()
println()
print("Enter the quiz name (without pdf extension): ")
quizName = readline()
println()
print("Enter the file containing student IDS (include extension): ")
idFile = readline()
println()
print("Enter the number of IDS: ")
numIDS = parse(Int, readline())
convertOptions = ["-density 200", "-crop 630x136+1010+80"]
convertFiles = [quizDirectory * quizName*".pdf", quizDirectory*quizName*"ID.png"]
#convertCommand = "-density 200 -crop 630x136+1010+80 " * quizDirectory * quizName*".pdf " *quizDirectory*quizName*"ID.png"
run(`convert -density 200 -crop 630x136+1010+80 $convertFiles`)
net = NNet.load_net()
processIDS(quizDirectory, quizName, numIDS)
saveGrades(quizDirectory, quizName, idFile, '\t', numIDS, 10)
println("Saved file $(quizDirectory*quizName*"grades.csv")")
