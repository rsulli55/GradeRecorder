# GradeRecorder

NNet.jl implements a neural network framework following the book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com)
  
graderecorder.jl has functions to process digitally scanned quiz papers to separate individual digits, and then pass these digits to a trained neural network to be predicted.  It then matches the predicted student IDs verse a known list and outputs a CSV file in the form: StudentID, Score
