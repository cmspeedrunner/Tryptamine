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
Lets just start with a hello world example, made fancy:
```rust
fn hello(){
  println("Hello, World!")
}
hello()
```
Pretty simple stuff, defining functions is done with `fn` and to print, simply use `println` or `print`. <br>
`println` -> Outputs value with a newline <br>
`print` -> Outputs value with no newline <br>

## Conditionals
Conditionals and loops are a bit different in Tryptamine. For example: 
```rust
number = int(read("Enter a number:"))
if number > 10{
  println("Number is bigger then 10")
}
```
This is simple. It sets the variable number to be an integer, read from the user. After this, it checks if the number is bigger then 10. If so, print something! <br>
*However* if we wanted to add an elif and else statement, it would look like this: 
```rust
number = int(read("Enter a number:"))
if number > 10{
  println("Number is bigger then 10")
elif number >= 0
  println("Number is between 0-10")
else
  println("Number is negative")
}
```
elif and else dont have opening and closing brackets in Tryptamine. <br>
An if statement must contain all conditionals that apply to it, for example this would produce an error in Tryptamine: 
```rust
name = read("Enter name:")
if name == "john" {
    println("Hi john, you're S tier")
} elif name == "mike" {
    println("Mike, I guess you're A tier")
} else {
    println("I don't know you")
}
```
`elif` and `else` do not need opening and closing brackets, only if does. <br>
This is how the code above will look in Tryptamine: <br>
```rust
name = read("Enter name:")
if name == "john"{
    println("Hi john, you're S tier")
elif name == "mike" 
    println("Mike, I guess you're A tier")
else
    println("I don't know you")
}
```
This is a bit disorienting for programmers, but is a unique and impactful addition of simplicity to Tryptamine, I completely understand any critique of this particular design choice, however I will say this becomes enjoyable and saves time when embedding loads of conditionals. <br>
*See `examples/` for more comprehensive examples.* <br>

## For and While loops
For loops are similar to python and other dynamic languages, here is a simple 1-100 program in Tryptamine using for and while loops: <br>
```rust
for i = 0 to 100{
  println(i) # Outputs numbers 0 - 99
}
i = 0
while i < 100{
  println(i) # Does the same as the above program
  i = i+1
}
```
You can also iterate over strings using `in`: <br>
*As of v0.1.2 you can iterate over strings only, sorry, I will update this soon*
```rust
string = "Hello, World!"
for char in string{
  print(char) # Prints each character with no newline, basically printing the string
}
```
True and false work like this: <br>
```rust
string = "Hello!"
fn isGreeting(string){
  if string == "Hello!"{
    return true
  else
    return false
  }
}

if isGreeting(string){
  println("It is a greeting!")
elif not isGreeting(string)
  println("Not a greeting!")
}
```
## Importing
Importing is done with `use` and the path is configured by default to be `std/`. <br>
***The majority of Tryptamine's power is from the standard library, make sure you have it*** <br>
For example, if we wanted to import the string library and use the `startswith` and `lower` function, it would look like this: <br>
```rust
use "string"
msg = "HeLLo there!"
msg = lower(msg) # makes all uppercase values lowercase
if startswith(msg, "hello"){
  println("Message is a greeting!")
else
  println("Message is not a greeting!")
}
```
This is just one example, the standard library is extensive and is covered specifically on the website. <br>

## Functions
Functions are defined like so:
```rust
fn FnName(p1, p2){
   return p1 + p2
}
```
But can also be done like this: 
```rust
fn FnName(p1, p2) -> p1+p2
```
You can have option parameters in a function by setting the value in the function definition:
```rust
fn add(num1, num2=false) -> num1+num2
println(add(5, 10)) 
println(add(5)) 
```
This program would output 15 and 5

## System 
The `system` function is inbuilt, being the sole inbuilt function to support the Axon package manager, the http standard library and the system standard library <br>
`system` runs a command as a subprocess, returning the value. For example:
```rust
hello = system("echo hello") # hello = "hello"
```
This program wouldn't output anything, it just stores the result of the system command in the `hello` variable. <br>
This is a quick example showing how this can be used to check the existence of a directory: <br>
```rust
fn succeed(command){
    status = (system(command+" >nul && echo 1 || echo 0"))
    return int(status)
}

command = "std\\"

if succeed(){
    println("You have the std library!")
else
    println("You do not!")
}
```
## Inbuilts
Here is a comprehensive outline of all the inbuilt functions and values in Tryptamine: <br>
`null` -> is null<br>
`false` -> is false<br>
`true` -> is true<br>
`_cwd` -> is the current working directory<br>
`_V` -> is the Tryptamine version<br>
`argv` -> is a list of arguments<br>
`println(a)` -> prints a with newline<br>
`exit()` -> exits program<br>
`system(a)` -> runs a as a subprocess<br>
`print(a)` -> prints a with no newline<br>
`date()` -> is the current date<br>
`time()` -> is the current time<br>
`read(a=null)` -> reads input from user, a is an option prompt to display<br>
`clear()` -> clears screen<br>
`isNum(a)` -> returns true or false if a is an int or float<br>
`isStr(a)` -> returns true or false if a is a string<br>
`isList(a)` -> returns true or false if a is a list<br>
`isFn(a)` -> returns true or false if a is a function<br>
`str(a)` -> returns a as a string<br>
`int(a)` -> returns a as an int<br>
`flt(a)` -> returns a as a float<br>
`list(a)` -> returns a as a list<br>
`split(a, b)` -> splits string a by occurances of b<br>
`stack(a)` -> splits string a into lines<br>
`clean(a, b=" ")` -> removes b from either side of a, defaults to whitespace<br>
`swapOut(a, b, c)` -> checks a for occurances of b, swapping all for c<br>
`rmPrefix(a, b)` -> removes prefix b from a<br>
`rmSuffix(a, b)` -> removes suffix b from a<br>
`append(a, b)` -> appends b to list a<br>
`pop(a, b)` -> pops value at index b from a<br>
`extend(a, b)` -> extends list a by list b<br>
`len(a)` -> gets length of list a<br>
`run(a)` -> runs a as a tryptamine file<br>
`openFile(a, b)` -> opens file a in mode b<br>
`readFile(a, b)` -> reads file a contents into bytelength value b<br>
`writeFile(a, b)` -> writes string b to file object a<br>
`closeFile(a)` -> closes file a<br>
`wait(a)` -> waits for number a as seconds<br>
