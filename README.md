# Tryptamine
Tryptamine is an dynamically typed interpreted programming language. I have attempted to make Tryptamine as freestanding as possible, everything that can be written in it, is. <br>
<br>
## Getting Started
To start, clone this repo using git and cd into it. <br>
At this point make sure you have g++ or any compiler for c++. <br>
`tryp.cpp` and `axon.cpp` are just routing files, these mean you can just run <br>
`tryp` <br> followed by your file to run, instead of invoking python each time, the same goes for installing packages with axon. <br>

Once both are compiled, you can start running Tryptamine programs, run the shell or install packages. <br>
To start the shell, just type:<br>
`tryp`, and it will start up the shell. <br>

To run a specific tryptamine file, you need to write one first.


# Tutorial
Tryptamine



# Axon
## Introduction
Axon is the package manager for Tryptamine, it is written all in tryptamine, it is under `src/axon.tr` <br>
Axon finds packages at https://github.com/cmspeedrunner/axon. You can create your own package by following the readme tutorial at the axon repo. <br>
## Usage
To install a package, given you have compiled the Axon.cpp file, simply type <br>
`axon install` followed by a package. Just for a test, you can do<br>
`axon install argparser` <br>
Which is a library I made for testing Axon. Following this you should see a folder in your `std` directory called `argparser` with the `main.tr` file under it. To include this in a project do: <br>
```
use "argparser/main"
```
*(if std isn't included in your path, this would be different)*
