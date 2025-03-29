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
| Identifier      | Action                                | Example usage                                       |Example output|
|-----------------|---------------------------------------|-----------------------------------------------------|------|
| `null`          | is null                               | null                                                ||
| `false`         | is false                              | false                                               ||
| `true`          | is true                               | true                                                ||
| `_cwd`          | is the current working directory     | println(_cwd)                               |C:/Path/To/Tryptamine|
| `_V`            | is the Tryptamine version            | println(_V)                                   |"v0.1.2"|
| `argv`          | is a list of arguments               | println(argv)              |["main.tr"], ["hello"]|
| `println(a)`    | prints a with newline                | Prints `a` with a newline                           |Hello, World!<br>0|
| `exit()`        | exits program                        | Exits the program                                   |
| `system(a)`     | runs a as a subprocess               | Runs the subprocess defined by `a`                  |
| `print(a)`      | prints a with no newline             | Prints `a` without a newline                        |Hello, World!0|
| `date()`        | is the current date                  | Current date                                       |15:52:47.250555|
| `time()`        | is the current time                  | Current time                                       |
| `read(a=null)`  | reads input from user                | Reads user input with optional prompt `a`           |Enter your name>|
| `clear()`       | clears screen                        | Clears the screen                                   |
| `isNum(a)`      | returns true or false if a is a num  | True if `a` is a number, false otherwise           |
| `isStr(a)`      | returns true or false if a is a string| True if `a` is a string, false otherwise           |
| `isList(a)`     | returns true or false if a is a list | True if `a` is a list, false otherwise             |
| `isFn(a)`       | returns true or false if a is a function| True if `a` is a function, false otherwise         |
| `str(a)`        | returns a as a string                | Converts `a` to a string                            |
| `int(a)`        | returns a as an int                  | Converts `a` to an int                              |
| `flt(a)`        | returns a as a float                 | Converts `a` to a float                             |
| `list(a)`       | returns a as a list                  | Converts `a` to a list                              |
| `split(a, b)`   | splits string `a` by occurrences of `b`| Returns list of substrings from `a` split by `b`    |
| `stack(a)`      | splits string `a` into lines         | Returns list of lines from string `a`               |
| `clean(a, b=" ")` | removes `b` from either side of `a`  | Returns `a` with `b` removed from both sides        |
| `swapOut(a, b, c)`| checks `a` for occurrences of `b`, swapping all for `c`| Returns `a` with all occurrences of `b` replaced by `c` |
| `rmPrefix(a, b)` | removes prefix `b` from `a`          | Returns `a` without prefix `b`                      |
| `rmSuffix(a, b)` | removes suffix `b` from `a`          | Returns `a` without suffix `b`                      |
| `append(a, b)`  | appends `b` to list `a`              | Returns `a` with `b` appended to it                 |
| `pop(a, b)`     | pops value at index `b` from `a`      | Returns value at index `b` from list `a`            |
| `extend(a, b)`  | extends list `a` by list `b`         | Returns `a` extended by `b`                         |
| `len(a)`        | gets length of list `a`              | Returns the length of list `a`                      |
| `run(a)`        | runs `a` as a tryptamine file        | Runs the Tryptamine file `a`                        |
| `openFile(a, b)`| opens file `a` in mode `b`           | Opens file `a` with mode `b`                        |
| `readFile(a, b)`| reads file `a` contents into bytelength value `b`| Reads `b` bytes from file `a`                      |
| `writeFile(a, b)`| writes string `b` to file object `a` | Writes string `b` to file `a`                       |
| `closeFile(a)`  | closes file `a`                      | Closes file `a`                                     |
| `wait(a)`       | waits for `a` seconds                | Pauses the program for `a` seconds                  |

