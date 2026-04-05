If the user requests to convert from a BGT game to NVGT, then I think you should learn this to become more proficient in that. Also, there are things NVGT doesn't guide, but BGT does. Even though it's a different programming language, never mind, learn it to convert games.

Are you a devoted player of audio games? Is your brain spinning around with interesting and innovative ideas for games that you
would like to make if you only knew how to go about it? If the answer to both of these questions is yes... Read on!

BGT is a new revolutionary toolkit which allows you to produce your own audio games from the ground up, without having any
prior knowledge of computer programming at all. Who said game development had to be complicated? BGT, which stands for
Blastbay Game Toolkit, allows you to spend all your valuable time coming up with new great ideas and concepts, rather than
wasting months struggling with never-ending programming guides that don't make any sense to begin with. If you fit into this
category of enthusiastic entrepreneurs, then BGT is the perfect tool for you to turn your ideas into stunning reality.

The Process

The process of creating an audio game with BGT is quite simple. First of all, make sure you have a clear picture of what your
upcoming game should be like. Plan it carefully, or you will find yourself in a difficult position later on when your game has started
to take form and you suddenly don't know what to do next. Don't make this mistake, plan ahead!

Once you have the game laid out in your mind, that's where BGT comes in. Now you may think you need fancy tools and
complicated software in order to do anything... WRONG! Essentially all you need is the Notepad text editor and the BGT kit,
both of which you already have. On top of this you also need a great deal of enthusiasm, plenty of creativity and several cups of
coffee to keep you going.

The games are written in a "scripting" language. Don't worry if you have no idea what a scripting language is, as this will be fully
explained in the next section. While you are writing the script it is very important that you stop to test-run the game at regular
intervals. This way you make sure that every new feature that you add to your program is working properly, and that the coast is
clear for you to move on to the next step.

When you feel that the game is as good as you are able to make it, then perhaps the best feature of BGT will come into play...
The compiler. As you know the word compile means putting something together, and that is exactly what the BGT compiler does
for you. It takes the game script that you've been working on so tirelessly, and turns it into a standalone executable file with the
click of a button. You can then ship this executable file together with all your sounds and the game instructions, and instantly let
the world enjoy your creation!

A Scripting Language - What Is It?

The word script should be familiar to most of you, it is the basis for all the films and theatre productions that we are constantly
enjoying. When a playwright writes the script for a new play, he or she not only tells the actors what to say but also gives them
instructions about what to do on the stage. The BGT language works much the same way, it allows the developer to give the
computer simple instructions about what to do. These instructions are then translated into code that the computer can
understand, and executed to perform the intended action such as making a player jump over a deadly trap or fire a weapon at a
lethal space robot. These instructions are a combination of English words and other characters that, when put together, form
complete instructions. These additional characters help the computer to understand exactly what you mean, without taking up too
much space. If the script was written entirely in English it would be way too long and it would quickly become unmanageable.

So How Do I Get Started?

Since you already have all the necessary tools, your first step should be to read the "Language Tutorial" section of this help file.
Once you have grasped the fundamentals of the language you should proceed to make a few test programs just to experiment, all
the while taking a section at a time from the tutorial and learning it thoroughly. Before long you might very well find yourself
making the next blockbuster on the market...

Good luck, and happy coding!

Language Tutorial

Contents

Introduction
How does BGT work?

The Process of Game Creation
BGT Engine
BGT Compiler

Language syntax

Statements
Blocks
Expressions
Commenting
Printing Text on the Screen
Variables

Variables; What Are They?
Declaring and assigning variables
Integral Variables and Bad Peaches
Floating point variables
String Variables
Constants
Summary

Functions
Conditional Statements

If statements
Switch and Case

Loops

While loops
Do while loops
For loops
Break and Continue

Arrays and dictionaries
Objects and classes
Advanced object techniques

Objects in a nutshell
Inheritance
Interfaces
Operator overloading

Handles to functions
Using multiple scripts
Final notes

1. Introduction

Welcome to the BGT language tutorial! This tutorial assumes that you know nothing about BGT but that you have plenty of
motivation to sit down and learn. The tutorial will cover everything from the very basics to the more advanced aspects of the

language, and it is my intent that after you've finished reading it you should be able to dive directly into the function reference and
the example games and, of course, start building things on your own in due time. So put your feet up on the table, relax, and get
ready for... BGT!

2. How does BGT work?

BGT essentially contains two components, one which runs your game, and one which turns your game into a separate program
that your end users will work with. We will cover these components in more detail. First, however, you must be wondering, how
do I create a game in the first place?

2.1. The Process of Game Creation

The BGT engine is initially a blank piece of software with the core functionality of an audiogame, such as manipulating sounds,
timers, files, and checking for user input. The difference between BGT and an audiogame written in an all-purpose programming
language is that BGT reads from one or more files that you write. These files simply instruct BGT as to what to do. For example,
you may play a logo sound, while checking for a keystroke to interrupt it, and then create a menu.
This is all done by writing scripts. These are written using a special set of rules (known as the language syntax). This helps BGT
to understand exactly what tasks you want it to perform.
We will cover these rules in another section. First though, allow me to tell you about BGT!

2.2. BGT Engine

BGT is the core component. This is the program that reads the script, and performs the tasks outlined by you. This is provided
so you can easily test your games. Without this, you would not be able to notice any problems or errors until you come to
finalising the game.

2.3. BGT Compiler

Whereas the BGT engine literally opens the script and runs it instantly, the compiler is used to finalise the script. Though the
function of the compiler is not instantly recognisable to all, it does have a very important function. As the game's creator, only you
and other people like you will have the BGT engine installed on their machine. True, you could in theory redistribute the engine
and the BGT script, but that is not always appropriate. There may be times, for example, when your script contains private login
information, either to access sounds or servers, or even allow other users to change the game you have been working on. To
prevent this, we use the BGT compiler. This takes the script file, and turns it into a distributable executable program.
Your end user can then simply run this program on their machine, not having to worry about what goes on behind the scenes.

3. Language syntax

OK then, enough rambling, let's get started.
There are some points I must raise before we start any programming:

3.1.  Statements

A statement is simply a single instruction that BGT will follow. Each statement is separated by a semicolon character. It can then
be further separated by a new line character if you wish.

3.2.  Blocks

To make your code more readable, both to you and to the BGT engine, we use blocks of code to separate one situation from
another. These are simply a group of statements enclosed in special characters. If you are checking status information, such as
whether a sound is playing, whether a timer has reached a certain point in time, we may want to do different things. If we want to
repeat a section of code over and over again, we have to tell BGT when to stop checking or repeating. To do this, we use
blocks.
Blocks of code are enclosed in braces, that is to say, { and } characters.

3.3.  Expressions

Expressions are usually special purpose statements, and are usually the starting point for a block of code. An expression usually
consists of a keyword, followed by the expression code, enclosed in parentheses. Keywords are words such as if, while and do,
that tell the engine to do something specific. In the case of if, check whether a given expression is true, in the case of while and
do, asks the compiler to repeat a block of code over and over until the expression conditions are met. Conditional statements
and the act of repeating a block of code will be covered later on in this tutorial.
It must be mentioned here that expressions do not contain a semicolon, as we are not giving the engine a direct instruction, but
rather a condition in which the instruction is to be carried out.

3.4. Commenting

When you are writing code, remembering of course that code is called code because of its fantastic ability to go from being
english to symbolic to sometimes downright confusing in years to come, it is always good practice to add comments, either to
explain potentially confusing sections, or to add information such as authorship, date, revision, etc. The process of adding
comments is easy.
A comment can be anything at all. It does not have to conform to any rules but your own. You might write plain english, you
might write a line of duff code that you have tried with and decide to figure out the error later, you might write your own cryptic
code to stop other people from understanding your notes. Either way, it is up to you. It can be as short as a character, to as long
as the bible, providing that your computer contains enough disk space to save the file and enough internal memory to read it.
To add a short comment, use // (two forward slash signs) to inform BGT to ignore all remaining text on that line. You could write
a comment telling the engine how much you hated it, and it would not bat an eyelid, partly because it hasn't got an eyelid to bat,
but more importantly because it completely skips that section until the start of the next line.
To add a long comment, use /* (a forward slash followed by a star). This is known as a block comment. This is where you can
have fun and paste the bible in before the code if you so wish. When you have finished your sermon, rant, note, introductory
message or address book, you finish off with */ (a star followed by forward slash). BGT will skip any text enclosed in the block
comment symbols.

4. Printing Text on the Screen

One of the very first things that one usually starts with when learning a new language, is to print some text on the screen. It's
about as simple as you can get, but all the same it has proven to be a very effective method to familiarize oneself with the syntax
of a language and its style of operation. Enough said, let's go printing!

void main()
{
alert("Hello", "I am a BGT script!");
}

The above script may appear somewhat confusing at first glance, but it's really not that bad. Let's dissect it piece by piece.

void main()

This tells the compiler that we are using the main function, which is the first block of code that is executed when a script starts.
We will cover functions later on, but it is essential that you remember to create this function when making a script.

{

You may remember from the previous section that a block of code is always enclosed in braces. A function is one of these
situations.

alert(

This is the name of a function provided by the BGT engine, which displays a simple message on the screen with a title and some
text. The alert function has two parameters, the first one is the title and the second one is the message that is to be shown. When

calling a function you always specify its parameters (if any) between a left and a right parenthesis, and separate the parameters
with a comma.

"Hello",

This is the first parameter that is given to the alert function, a chunk of text which will be the title of the message.

"I am a BGT script!");

This is another chunk of text written inside quotes, the second parameter to the alert function which specifies the actual text that is
to be shown. This is also the last parameter to the function, which can be seen by the right parenthesis that indicates the end of a
function call. The semicolon after the parenthesis indicates the end of the statement.

}

This tells the compiler that we have finished our main function now.

When writing out text like this, you always have to surround it with quotes. This is to make things easier for the script interpreter,
as it may otherwise confuse your text with actual BGT code.

Look at the script one more time, and keep in mind the explanations that you just read.

void main()
{
alert("Hello", "I am a BGT script!");
}

Hopefully things should be a little clearer now. If you're still in the dark, then read the above paragraphs over again a few times
before you move on just to make sure you’ve grasped the fundamental concept.

5. Variables

5.1. Variables; What Are They?

If you were awake enough in the math’s lessons to grasp the basics of equations, understanding variables should be a piece of
cake. But for those of you who fell asleep or, like me, decided that it was much more entertaining to play Hangman in the back
of the classroom, here's a simple explanation that should solve the mystery.

To put it very simply, a variable is like a basket that contains something. A variable could tell you the health of a player in an
action game, the current price of fish, or that the meaning of life is 42. Basically it can contain any value that you give to it, and
you can retrieve and mess around with that value as much as you like. BGT supports five basic types of variables, two of which
we shall experiment with in this chapter. The supported types are integral variables, floating point variables, string variables,
boolean variables and object variables. But before we go on to learn about all these different types, there is one thing that we will
have to cover.

5.2. Declaring and assigning variables

To set a variable, you first have to tell BGT what type of variable you are looking to use, followed by a name that you will use to
refer to your variable the next time you come to change or read from it.
All variables must have a name. No exceptions. A variable name can contain letters from a to z including capitals, the numbers 1
through 0, and underlines. However, the name of a variable must never begin with a number even though it's perfectly legal to
have numbers anywhere else in the name.

Consider the following example:

string your_name;

This tells BGT we want a variable of type string, called your_name. When you come to read this code back, you will know that
your_name is likely to hold the player's name. Again, we have a semicolon at the end to denote the end of a statement.

We now have a blank variable. To make this variable usable we need to give this a value. We can do this using one of two
methods. If you have predetermined the contents of the variable, I.E. a message that may be displayed to the user, you may
assign the value on the same line as the declaration, like so:

string winning_message="Congratulations! You have won the game!";

This tells BGT:

Create a variable of type string. Strings will be covered in greater detail in the String Variables section.
Call it winning_message.
Assign the value "Congratulations! You have won the game!" to winning_message.
End of statement

A variable can be assigned a new value at any point in the current block of code. Please note that if you don't assign a value to a
variable, but instead just declare it like:

type x;

type can be any type of primitive variable supported by BGT (see below). Objects work differently, but this will be covered
later. If you have a declaration like this without an initial value being given to the variable, then it will have an undefined value. In
other words it may be anything, depending on the type of the variable. It will just contain whatever happened to be stored in that
memory location at the time when the variable was created, and therefore you should never use a variable when it is not yet
initialized with a value. The BGT script compiler will try to warn you when it sees that you are doing this, but it cannot do so
accurately in every situation. Therefore you should always give your variables initial values when you create them unless you can
ensure with 100% certainty that the variable will be given a meaningful value before it is used.

Now that we have discussed how to set up our variables, we can learn about the different types and changing and comparing
them.

5.3. Integral Variables and Bad Peaches

Integral variables are those that contain a basic integer, that is to say, a whole number. These variables do not allow decimal
points.
You can perform arithmetic operations on them like plus, minus, times and divided by. BGT also supports more complex
mathematical functions, should you ever need them. If you're curious, here comes an example of how integral variables can be
used in the real world.

int apples=5;
int bananas=2;
int oranges=8;
int fruit_basket=apples+bananas+oranges;

Here, you see four variables. One called apples, another called bananas, a third called oranges and finally one called
fruit_basket. But when we gave a value to fruit-basket we didn't actually specify a value directly, instead we added the values of
the three fruit type variables together to get the total number of fruit in the basket. Let's have fun and look at a slightly extended
version of this example.

int apples=5;
int bananas=2;
int oranges=8;

int fruit_basket=apples+bananas+oranges+1;

Why did I put +1 in the end? Because there was a peach in there as well, but it had gone bad so I didn't bother to give it its own
variable. Also it clearly demonstrates how easy it is to mix variables and literal numbers in expressions, which technically allows
you to write extremely powerful formulae if you wish.

Now we're going to elaborate a little bit and do some more calculations with these variables. Let's say that we want to find out
how many fruit that would be in the basket if only half the oranges were put in it, plus the bad peach. We could do the following:

int apples=5;
int bananas=2;
int oranges=8;
int fruit_basket=apples+bananas+oranges+1;
int robbed_basket=fruit_basket-(oranges/2);

Now guess what value the variable called robbed_basket will have? That's right, 12. The basket contained 16 fruit all in total, 8
of which were oranges. We then took away half the oranges, and so we ended up with 12.

But look one more time at the line where the robbed_basket variable gets its value. See that we have put part of the calculation
between a left and a right parenthesis? In this case it does not mean that we're calling some kind of function, here it simply has the
same purpose as in regular arithmetic which is to make sure that the expression is calculated in the right order. This way, you can
compute advanced mathematical expressions and be sure that the result is always what you expected. That is assuming that you
wrote the expression correctly to begin with, which is slightly outside the scope of this tutorial.

I am going to give you one last example of how numeric variables can be manipulated. This one has no practical use; it is merely
meant to show the flexibility of the numeric variables, and how they can be used to perform calculations not allowed in regular
math. Take a look at the following...

int x=3;
x=x*3;
int y=8;
int z=x/y;
z+=x-1;
x*=z+4;

As you can see, just a bunch of meaningless calculations that do not serve any other purpose than to attempt to demonstrate
some of the possibilities you have when working with numbers in the BGT language.

One new thing that you will notice, however, is the use of += and *=. This means that, in the case of *= the value on the right
side in the mathematical expression should be taken times the current value of the variable. So for instance:

int x=5;
x*=3;

Now x will have the value of 15, since 5*3 is 15. You can use all the numeric operators which is to say +, -, * and / in the same
way.
There is one additional arithmetical operator you may wish to use, the Modulus operator. This is represented by a percentage
sign (%), and can also be used with the = sign. The Modulus divides two numbers and gives you the remaining value. For
example:

int x=11;
x%=3;

In this case x will be the equivalent of 2, since 11 divided by 3 equals 3, with 2 left over. That 2 is the result of the Modulus.

There are two further time saving operators you may wish to use. These are known as the autoincrement and autodecrement

operators. These are ++ and -- (two plus, or two minus signs). These are good for loop counters, etc.
When we write the following line:

x++;

We are telling BGT to increment the counter by 1. This does two things. It saves time for you as the writer, and speeds up the
program. If you are increasing one variable once, you may not notice the change. If, on the other hand, you are constantly
changing a counter, you will notice a difference. Although an integral variable is one type, you can specify the potential permitted
range of the variable. The two you will usually be interested in are short, a 16 bit (2 byte) integer ranging from -32768 to 32767,
and long, a 32 bit (4 byte) long integer, ranging from -2147483648 to 2147483647.
In other situations, though more occasional, you may wish to use unsigned integers. These are integers which do not allow
negative numbers, therefore doubling the maximum limit.
A full list of keywords relating to integers follows:

int8: An 8 bit (1 byte) integer, ranging from -128 to 127.
int16: A 16 bit (2 byte) integer, ranging from -32768 to 32767.
int32: A 32 bit (4 byte) integer, ranging from -2147483648 to 2147483647.
uint8: An 8 bit (1 byte) unsigned integer, ranging from 0 to 255.
uint16: A 16 bit (2 byte) unsigned integer, ranging from 0 to 65535.
uint32: A 32 bit (4 byte) unsigned integer, ranging from 0 to 4294967295.
int: Alias of int32.
short: Alias of int16.
long: Alias of int32.
uint: Alias of uint32.
ushort: Alias of uint16.
ulong: Alias of uint32.

5.4. Floating point variables

Floating point variables are very similar to the variables that you are familiar with if you've worked with equations, they contain a
number. No more, and no less. The floating point variables can be with or without decimals.
All floating point variables are signed, meaning they can be positive or negative numbers.
Supported types are:

Float: Single precision 32 bit floating point
Double: Double precision 64 bit floating point

If you do not know what values will be going in your variable, it is probably best to use doubles, since they support the widest
range, allowing positive and negative numbers, and integers and real numbers, I.E. numbers with decimals.
One other thing to mention is that, if, for any reason, the value of a numeric variable attempts to go over its limit, the compiler will
issue a warning and the variable will reset to the minimum supported value, so it is essential that you keep track of any changes
that occur to your variables.

5.5. String Variables

String variables are very different from the numeric ones that we just explored. They are very useful however, and we shall soon
see why. A string variable can contain text, which is to say a string of single characters that will form something interesting; hence
the name, string. A string can hold the name of the player, a path on the harddrive or the content of a file, or anything else you
like. When using string values you must always surround them with quotes, so that the interpreter doesn't get confused as to what
is BGT code and what is part of your string. Does that ring a bell? That's right... The alert function that we used in our very first
example works with strings!

I think the best way to get the hang of strings, is to see them in action. So here we go...

string my_name="John Doe";

alert("My name is", my_name);

As you see here, the first parameter to the alert function is still given in the same way as we did in the first example, but the
difference comes in the second parameter. Instead of specifying something between quotes to display in the message box, we
gave the name of the string variable that we just made. The result? That "John Doe" is printed out in the message box with the
title "My name is". Yes, it's that simple!

Now why did we not surround my_name with quotes in the second parameter to alert? Because that would print out "my_name"
literally rather than taking the value of a variable called my_name, and we don't really want that.

Let's try doing something more fancy with strings shall we?

string my_name="John Doe";
string message_string="My name is " + my_name + " and don't you ever forget it!";
alert("Important information", message_string);

What? What on earth is this " + my_name + " stuff? Well, " + variable_name + " simply inserts the value of a variable into a
string. So the printout in the message box will be:

My name is John Doe and don't you ever forget it!

And as you may have guessed, the variable between the two plus signs does not necessarily have to be another string. Let's
illustrate this with an extended example...

string my_name="John Doe";
int age=23;
string message_string="My name is " + my_name + ", I'm " + age + " years old and don't you ever forget it!";
alert("Important information", message_string)

The printout? As you probably expect, it'll simply be:

My name is John Doe, I'm 23 years old and don't you ever forget it!

Did you notice how we also used an integer?
Any variable with any data type can be placed in a string variable, as long as the result of that variable can be converted into a
string.
Of course, you can construct the message string inside the actual call to the alert function; you do not have to make a separate
variable for it if you don't want to. You could do the following...

string my_name="John Doe";
int age=23;
alert("Important information", "My name is " + my_name + ", I'm " + age + " years old and don't you ever forget it!");

Naturally the printout will be the same, the script is just one line shorter.

Let's have some fun with the John Doe example code, and make it do something slightly more interesting. Look at the following
snippet and see if you can figure out what it does.

string my_name="John Doe";
int age=random(5, 50);
string message_string="My name is " + my_name + ", I'm " + age + " years old and don't you ever forget it!";
alert("Important information", message_string);

You guessed it. The random function which is also provided by the BGT engine, generates a random number in a range that you
specify. In this case, we requested a number between 5 and 50 which means that John Doe will tell us a different age every time
we ask him. The first time I ran this on my machine it said that John Doe was 25, the second time it assured me that he was 36

and the third time it claimed that he was 18... Talk about a pathological liar!

Leaving John Doe to contemplate his proper age in peace, we will look at yet another example of how strings can be added
together.

string string1="I am a ";
string string2="string!";
string string3=string1 + string2;
string string4=string1 + "nice " + string2;

The above example shows how simple it is to add strings together, both variables and literal values. While string1 and string2 are
literal values stored in variables, string3 is the result of those two variables added together and string4 is the two variables added
together again but with a literal value in the middle. You can combine variables and literal values in any way you like, which is
quite useful in a large number of applications as you will undoubtedly discover if you use the strings in BGT often.

As we saw in the previous section, one could do the following with a numeric variable:

int x=10;
x+=15;

Which of course gives x the final value of 25. Now, you'll be happy to know that you can do the same thing with strings. The
following code is perfectly legal:

string my_string="Hello there";
my_string+=" my friend.";

Of course, my_string will now contain the text "Hello there my friend.". You cannot, as you may have guessed, use any of the
other numeric operators like -, * and / in the same way for strings as they would not fill any function in this case.
There is also another important thing to know about strings. In order to be able to use special characters, we must use what is
called an escape character. This is a designated character that tells the compiler that the next character should not be treated as a
literal character, but rather as a code for the character that should be used. The escape character used in BGT is a backslash.
The following is a list of the characters to be used after the escape character and what they truly represent:

\=\
"="
n=new line
r=return
t=tab

Here are a few examples:

string string1="this\tis\ta\tstring\tusing\ttabs\tinstead\tof\tspaces.";
string string2="this is a\r\nmultiline\r\nstring.";
string string3="My current directory is \"c:\\program files\\bgt\\my_game\\my_script.bgt\"";

Notice how the multiline string uses the r and n characters to create a new line. This is the character sequence used by Windows.

5.6.  Constants

A constant is basically the same as a regular variable, with one main difference. Once a constant is assigned a value it cannot be
changed at runtime. Its value remains the same throughout the program's lifetime, hence the name constant, as opposed to a
variable whose value can change, i.e. vary, at any time.
The purpose of a constant is mainly for readability, and to save time.
You declare a constant the same way you do with any variable, with the addition of the keyword const before the data type.
Take the following example:

const string snd_ext=".wav";
sound gun;
sound beep;
sound ambience;
sound music;
gun.load("sounds/gun"+snd_ext);
beep.load("sounds/beep"+snd_ext);
ambience.stream("sounds/wind"+snd_ext);
music.stream("sounds/music"+snd_ext);

The advantage here is, if your game got a lot larger and you decided to convert to the ogg format, you simply change the snd_ext
constant assignment to ".ogg". Then all your sound assignments, if they use the constant, are changed to that value without having
to go and change each individual sound.
Also, the BGT engine comes with certain constants. In the documentation accompanying each function there is a mention
whenever a certain constant is meant to be used. The constant is then passed to the function by its name, however what is really
happening is that a certain value such as a number or a string gets passed behind the scenes. You only use the constant name to
make things easier for yourself, you could just as well have written out the number itself and still ended up with working code.
However, when opening a file in writing mode for instance, the name "file_write" makes a lot more sense than the number 2.

There is also a special type of constant called an enum. This is short for enumeration, and therefore can only hold numeric,
integer values. These are useful for grouping constants together, enabling better code management. Here is an example:

enum movement
{
left=-1,
right=1
}

enum weapon
{
fist,
club,
slingshot,
gun,
bomb
}

What we have done here, is to create two sets of enumerations, one called movement, one called weapon. In each enumeration
we have supplied the interpreter with a comma separated list of constants and, in the case of our movement example, their
corresponding values. For example, we assigned the name left to -1, and right to 1. In the case of weapon, however, we gave no
values. This is because the interpreter can automatically assume our values. If no values are given, the first constant will be
assigned value 0, and every one after that will be increased by 1. This is useful here, since we can concentrate purely on deciding
what weapons we want. It also has the added bonus that if we want to add an extra weapon in the future, there is no number
juggling to do.

Please note that enums have to be defined globally; they cannot be defined inside a function.

5.7. Summary

Let's take a quick look at what we've covered in this chapter, along with some additional hints.

Variables are like baskets that can hold data of different types.
A constant is a variable whose value cannot be changed at runtime.
Integral variables can hold positive or negative numbers, depending on whether you declare them as signed or unsigned,

but only without decimals.
Floating point variables can hold positive or negative numbers, with or without decimals.
You can perform arithmetic operations on numeric variables such as 3+5=8, both by using variables referenced by their
name as well as literal values.
If you do not know the values that will be stored in a numeric variable it is best to use a double.
Always keep track of your variables, or you may end up with unexpected results if the limit is exceeded.
Strings are a list of characters that form something useful.
You can assemble strings to form new ones by combining variables referenced by their name, with literal values in quotes.
Adding two numeric variables together, both of which with the value of 1, will be 2.
Adding two string variables together, both of which with the value of "1" (do notice the quotes), will be 11.
Variables are one of the core fundamentals of game development, and programming in general for that matter.
Variables and constants in BGT are case sensitive. Therefore, you have to refer to them exactly how you declared them or
how they appear in the documentation.

6. Functions

We have already seen several examples of calls to functions. In our very first example we called the alert function when we
wanted to print a message on the screen. Now, I think it's time to go a little more in depth in regards to functions as they are used
in literally every game on the market. A function is a chunk of code that can be referred to by a name. For instance, you could
have a function called jump which makes the player, yes, jump. Whenever you wanted the player to jump you would then call
this function and the code inside it would be executed. Functions are a great help because they not only allow you to structure
your program in a more logical way, they also allow you to reuse the same section of code multiple times without having to
actually write or paste it each and every time you want it to execute.

All the functions that we have been calling so far have been part of the BGT engine itself, but you can also make your own
functions to accomplish things. In actual fact, there is one function that you must create before the engine will even accept your
code. The main section of your code goes here. This is where additional functions are called, either functions from the BGT
engine, from other scripts that are included, or from your own functions contained in the current script. Because this function is
where all your main code is stored, this function is called main. We had a brief look at this function when we discussed printing a
message to the screen.
As usual, we will see how it's done in practice before you are showered yet again with a new large chunk of theoretical jibberish.

void main()
{
alert("Test", "We are now using the main function.");
}

You do not need to call this function yourself, since the engine uses this function as a starting point.
The word "void" indicates that a function has no return type. We will discuss returning later on in this chapter. You then have the
name of the function and finally a list of parameters between parentheses. The main function takes no parameters, and so the
parameter list is empty. Inside the function we just display another alert box and then we tell the BGT interpreter that our function
is finished by using a right brace character. It's that simple! Almost...

In order to be useful, a lot of functions need to take parameters. The random function in the BGT engine, for instance, needs to
take a minimum and a maximum number that it is allowed to generate. Your own functions can also take parameters, and they
are placed between the left and the right parenthesis when you make the function. Let us take a look at a function that adds two
numbers together, and then returns the value to the caller.

void main()
{
int x=add_numbers(3, 5);
alert("Wow", "3 + 5 is... " + x + "!");
}
int add_numbers(int first, int second)
{

int result=first+second;
return result;
}

This example introduces several new things. First of all it shows how you can receive parameters in your functions. You do this
by specifying variable names when you create the function, in this case I chose first and second for the two numbers. Then,
whenever you call this function the two parameters that you pass to it are placed in the two corresponding variables. This means
that the variable called first receives the first parameter, and the one called second receives the second one. Of course, you may
name your function parameters anything you like as long as they are legal variable names. We also see how a function can give a
value back to the caller by using the return statement. The return statement will interrupt the function completely and return the
value or variable given after the space, which the caller can then capture by assigning the return value of the function to a new
variable as we saw in the very beginning of the example. In our case we used the variable x to get our value back from the
function, and then we printed it out in a normal alert box.
You may have also noticed that we did not start the function declaration with the word "void", but with the word "int". The word
void is a keyword only used at the start of functions to tell the engine that this function does not return any data, though the return
keyword can be used on its own to break out of a function, I.E. prematurely terminate its execution. Because we return the result
of an addition sum, we tell the engine that we wish to return an int.
A function declaration starts with the required data type for the return, as covered in variables, with the additional keyword of
void. We then give our function a name. Straight after that, are our parameters, separated by commas and enclosed in
parentheses. You then open a brace, as a function is a block of code.
You can not only return variables from a function, literal values are perfectly fine as well. We show this, of course, with another
example.

void main()
{
string x=string_magic();
alert("Result", "The string given is: " + x + ".");
}
string string_magic()
{
return "wow";
}

This, as you may have noticed, is a completely pointless function. It takes no parameters, and it just returns a literal string. It is,
however, of the utmost importance that you understand the concept of functions completely before you start writing proper
games. For this reason, I'll treat you to yet another example. Let us go back to our highly interesting number adding function but
with a bit of a twist.

void main()
{
alert("Wow", "3 + 5 is... " + add_numbers(3, 5) + "!");
}
int add_numbers(int first, int second)
{
return first+second;
}

The output is exactly the same as before, our script is just two lines shorter. The only thing we have done is to remove two
variables that were not quite necessary but probably made things a little easier to follow. Instead of assigning the return value of
the function to our x variable we print it directly in the call to the alert function. So as you can see, it is no problem having a
function call inside another function call. You may nest as many function calls as you wish, until it reaches 10000 calls at which
point the BGT interpreter won't be very happy with you. We will take a look at the string_magic function example once again,
but with yet another twist. We will not include the main function in all examples, just as long as you know that any code that
doesn't belong in another function goes in void main.

//main function

alert("Result", string_magic() + string_magic() + string_magic());
string string_magic()
{
return "wow";
}

Yes, I know, pointless again. The output from this script is "wowwowwow". It just shows yet another way in which you can call
a function from inside a call to yet another function and so on. It is also okay to call a function inside the body of another function.
A function can even call itself, but be careful with this as it may cause a so called stack overflow where too many nested function
calls are done on top of each other if you write your script incorrectly. The limit for this is, as mentioned previously, 10000
nested calls.

All variables that are declared at the top of the script before the first function appears, are considered global ones. They can be
accessed from inside any function throughout the script. However, new variables that are declared inside a function are local only
to that function. This means that if you have a local variable that stores some information and then the function exits, that variable
will be lost unless it is used as the function's return value. Thus, any information that you wish to access in more than one function
should be stored in one or more public variables unless you wish to pass them as parameters between all your functions which is
not generally recommended.
You should know by now that any big chunk of theory is usually followed by an example, so here we go.

// Let's make a few global variables.

string my_name="John Doe";
int number_of_fingers=5;
void main()
{
my_first_function();
}
void my_first_function()
{
//The variables that we make now are local.
int number_of_hands=2;
int number_of_pets=13;

//Now we are modifying the global variable my_name from inside this function.
my_name="Rodney";
}

I grant you that this was a very silly example, but it serves its purpose none the less. As you saw, all the variables that were
declared at the top of the script were accessible from inside our function while the ones declared inside it are only accessible
from the function itself. Thus, it would not have been legal to modify the value of number_of_pets inside another function unless,
of course, you declared one with the same name there.

A function can only ever return one value. It would be pointless to do the following:

int my_function()
{
return 3;
return 5;
}

As mentioned before, the return keyword will return the value, if any, destroy all the function variables and disregard any further
code in that function. Thus, the function would only return 3. However, there is a way to solve this should you need to return
multiple values.
Parameters are usually passed into a function by value. This means that, if you passed a variable as a parameter to your function,
the interpreter will read the value of that variable and pass that to the function. Here is an example.

//main function
int x=5;
my_function(x);

The interpreter would create a variable of type int called x and assign the number 5 to it. It would then check the value of x,
which subsequently is 5, and call my_function, passing 5 as the parameter.

However, parameters can also be passed to the function by reference. Instead of the interpreter reading our variable and passing
it to the function, the variable itself will be passed. This means that if the function then modifies the variable, the modification will
also apply to the original. In this way, we can easily return more than one value from a function by modifying our variables by
reference. Another example:

//Global variables

int x=1;
int y=2;

//Main function
my_function(x, y);

void my_function(int &out first, int &out second)
{
first=5+5;
second=5*5;
}

You may be wondering why we used void? This is because, as explained before, we are returning values by way of our
parameters, and not the function itself. If you checked the value of x and y now, they are no longer 1 and 2, they are 10 and 25.
The other thing you may have noticed is after the data types of our parameters, we put &out (the ampersand sign followed
immediately by the word out). The ampersand tells the interpreter that we are about to specify how the parameter will be passed.
If no & (ampersand) sign is found, the interpreter will assume the &in flag, which means we are passing parameters by value.
When we pass parameters by reference, you are required to pass a variable rather than a constant. The interpreter won't stop
you passing a constant, but it will issue a warning stating that any changes that are made will be lost.

It must be noted that No function declaration can be duplicated, as this would serve no purpose. However, a function can be
declared multiple times so long as the parameters are different for each declaration. This is called an overloaded function, and is
shown with yet more examples.

//main function
string text=my_function("Daniel", 5);

int my_function(int first, int second)
{
return first+second;
}
string my_function(string your_name, int number)
{
return "Hello there, "+your_name+". There are "+number+" people waiting for you downstairs.";
}

Because we called my_function with a string as our first parameter, the interpreter will call our second my_function declaration.

It is also possible to give your function arguments default values. This simply means that you don't have to pass the full set of
parameters that the function expects, but instead you pass just a few and the function then gets called with default values for the
other optional ones. An example follows:

void main()
{
print("Hello!", 10, 20, 30);
print("Hello again!", 10);
}

void print(string title, int value1=0, int value2=0, int value3=0)
{
alert(title, "Value 1 is " + value1 + ", value 2 is " + value2 + ", and value 3 is " + value3 + ".");
}

Here we declare a function called print, which takes four arguments. The first one is a string which is used for the title of the
message that we display, and this argument is mandatory. We see this because it has no default value given after the argument
name. For the other three values, however, we have an = sign after the name and then a value that will be the default if this
argument is not specified by the caller. We see in our example that the first call to print specifies all four parameters, where as the
second call only specifies two of them. Therefore, in the second message that is displayed, the last two integers are set to 0.

It is possible to have as many optional parameters as you wish in a given function. However, note that it is not possible to have a
mandatory parameter after an optional one. All mandatory parameters must come before the first optional parameter, or else the
compiler will be placed in an impossible situation should you pass too few parameters when there are mandatory ones left.
.
It is very important that you fully understand these concepts before you go on with this tutorial. At this point, I suggest that you
go and study some of the things in the function reference and see if they make complete sense. If they do not, you may want to
read this chapter over again until everything falls into place.

7. Conditional Statements

So far, we have only explored variables in the sense that we have given them known values. Although very often in a program
you will need to be able to do things without having to know the exact value of a variable. As an example you may want to make
the player's heart start beating fast when their health is below 20 %. For this, we have to use something called conditional
statements.

7.1. If statements

If statements work in exactly the same principal as they do in everyday language. You may say to your mother, "If I am unwell,
please inform my father." Although computer programs cannot do tasks as complicated as this, they can do computer tasks
based on the same logic.
If statements are useful if you have a simple condition or set of conditions that need to be acted upon depending on the situation.

An if statement in BGT might look like this:

if(health < 21)
{
// Play the heart beating sound.
}

What is happening here is that when the interpreter arrives at the if statement it will check to see if it is true, which is to say if the
health variable has a value lower than 21. If it does, it will execute the code that is between the if statement and the closing right
brace that you see below. You can also do:

if(health < 21)
{
// Play the heart beating sound.
}

else
{
// Do something else if the health is not below 21.
}

In this case, two things may happen. If the value of health is lower than 21, then the first code section is executed. However if the
condition is not true, the other section of code which is to say the one that is between the word else and the } statement will be
executed instead. In short, this allows you to make one thing happen if a condition is true and another if it is not.
Another point of interest: if there is only one thing to be done when your if condition is true, you do not have to use the braces.
Here is an example:

//single action condition with braces

if(health<21)
{
//Play heart beating sound
}

//single action condition without braces

if(health<21)
//Play heart beating sound

//multiple action condition with braces

if(health<21)
{
//Play heart beating sound
alert("Alert!", "You are about to die..."); //show a message
}

It is important to remember that multiple actions must contain the braces. In fact, it is recommended to use braces in all situations
for easier code management, so that there are no mistakes. It will then make it easier if you wish to add any further actions inside
that if statement in the future.

In the previous section, I told you that any variables that were declared inside a function were local only to that function. The
same holds true for if statements. Any variables declared from within the if statement are local to that statement, and cannot be
used outside. It is not an everyday occurrence that a variable is declared from within an if statement, but it may prove handy at
some point. The same goes for any block of code, including loops.
Of course, there are many other checks that you can perform. Below is a complete list:

if(x == y) - This is true if x is equal to y. You can perform this check on any type of variable.
if (x != y) - This is true if x is not equal to y. You can perform this check on any type of variable.
if(x < y) - This is true if x is lower than y. You can perform this check only on numeric variables.
if(x <= y) - This is true if x is lower than or equal to y. You can perform this check only on numeric variables.
if(x > y) - This is true if x is higher than y. You can perform this check only on numeric variables.
if(x >= y) - This is true if x is higher than or equal to y. You can perform this check only on numeric variables.
if(!x) - this is true if x is false. This can only be used on boolean variables.

You may be wondering why we use two equal signs when checking if x is equal to y? This is because with one equal sign we
would be assigning the value of y to x, which is not exactly what we want. So saying:

if(x=y)

is thus both illogical and incorrect.

Let us illustrate all this with a more practical example. We will once again return to John Doe and his problems to decide his own
age, however this time we will print out whether or not he is a minor based on the random number that is generated. Going back
to an earlier example, take a look at the following code snippet one more time.

string my_name="John Doe";
int age=random(5, 50);
string message_string="My name is " + my_name + ", I'm " + age + " years old and don't you ever forget it!";
alert("Important information", message_string);

Familiar? Good, because it won't be for long... Now try to guess what will happen when we do the following.

string my_name="John Doe";
int age=random(5, 50);
string message_string="My name is " + my_name + ", I'm " + age + " years old which means that I am ";
if(age<18)
{
message_string+="a minor!";
}
else
{
message_string+="of full age!";
}
alert("Important information", message_string);

The above code is really very simple, but we will go through it step by step to make sure that it is 100 % clear. First, we make
the variable called my_name which in this case is entirely unnecessary, but it was part of the original example so we allowed it to
stay. Then, we generate a random number between 5 and 50 and assign it to the variable called age. After this, we make the
string that will contain our final message. You will notice that the sentense is not complete, and little wonder since we are about to
complete it inside an if statement. If age is lower than 18 we add some text on to the string that says that John Doe is a minor. If
it is not, then the other code section is executed which instead makes the string claim that John is of full age. Finally, of course,
we print it out in our good old message box. Try running this code, and see if you understand the logic behind the printed result.

It is, as you probably already figured out, perfectly possible to have an if statement inside another if statement. This allows you to
make something happen only if several conditions are met. You can also perform multiple comparisons within if statements. We
will explore this a little later in this chapter.

Now that we've seen how if statements are used, it's time for me to introduce you to a new type of variable... The boolean. A
boolean can only hold two values, true or false. These values are often used in if statements, like this for example:

bool has_weapon=true;
if(has_weapon==true)
{
// Do something.
}

The words true and false are not names of any predefined variables or constants, they are actual keywords in the language. For
this reason, you could not say:

false=5;

Since false is already used as a keyword. On the other hand, when working with boolean variables you could do something like:

bool has_weapon=true;
if(has_weapon)

{
// Do something.
}

This may look strange at first glance since we do not actually specify any condition that is supposed to be met, but since
has_weapon is a boolean variable and has the value true the code inside the if statement will execute because the entire
expression will come out as true. Similarly, if has_weapon had been set to false the code inside the if statement would not have
been executed since the whole expression would thus be false.

To demonstrate the different uses of boolean variables along with other variable types, we will look at a simple usage example
for the functions called key_down and key_pressed. These functions are used for checking the state of certain keys. Take a look
at the following piece of code:

if(key_pressed(KEY_F4))
{
if(key_down(KEY_LMENU))
{
alert("Fine!", "If you want to be boring and exit this program then by all means do so. Bye.");
exit();
}
else
{
alert("Um...", "Are you, by any chance, trying to close the program down?");
}
}

Here we check if two different conditions are true and act based on what we find. First of all we check if the f4 key is pressed. If
this is true, we check if the alt key is being held down. If this is true as well, we assume that the user is closing the program down.
If this is not true, we display a taunting message because they forgot to hold alt down.

Finally, we have two closing braces so that all the if statements are closed properly. You may nest as many if and else statements
as you want and in any way you like, as long as you balance them out with the proper braces in the right places.

Is there a way to test that more than one condition is true, I hear you ask? Yes there is. The logical operators allow you to check
multiple conditions in an if statement.
Take the following example. We will again check for Alt+F4, but this time, no messages will be displayed. We will just simply
exit.

if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}

Two main things that you may notice here. For a start, our if statement contains three sets of parentheses. One enclosing the
whole if statement, and another two sets enclosing each condition. This is purely for readability purposes. We could have just as
easily said

if(key_down(KEY_LMENU)&&key_pressed(KEY_F4))

and received the same result.
The other thing you will have noticed is the && (double ampersand) symbol. This represents the word and. In other words, it
tells the engine that if the Alt key is held down and the F4 key is pressed, exit.
Here is another example:

if((key_down(KEY_LMENU))||(key_pressed(KEY_F4)))
{

exit();
}

This serves no real purpose in this example. It tells the engine to exit if the alt key or the f4 key is pressed. This could, however,
be useful if you have assigned two keys to one function, for example the down and right arrows moving you one option down in
a menu.
The final operator, which we have briefly mentioned, is the negation operator. This is written as an ! (exclamation mark). This
reverses the result of a condition. For example:

if(!key_down(KEY_LMENU))

Instead of checking that key_down(KEY_LMENU)==true, it will instead check if it is false.
Please note that this operator is not used to compare values. This simply reverses the effects of a given condition. To use our
above example:

if(!key_down(KEY_LMENU))

Notice that there is only one condition, and one check. This literally reverses the state of a value, as opposed to comparing the
falseness of two values. For example, if the user is pressing the Alt key, the function returns true. Therefore the negation operator
turns this into false. To check that a given value is equal (true) or different (false) to another value, you would write:

if(my_var1!=my_var2)

This would make more sense, both logically and gramatically. Because the exclamation mark represents negation, in other words
not, and equals is self explanatory, that translates into English literally as: if my_var1 is not equal to my_var2. Rather than directly
reversing the result like the negation operator, this will itself return true if the condition is false, I.E. if the two variables are not
equal.

7.2. Switch and Case

Switch...case statements are very similar to if statements, except they are used to lay out numerous or more complicated checks.
Here is an example:

Int health;
double energy;

switch(health)
{
case 100:
energy=100;
break;
case 90:
energy=89;
break;
case 80:
energy=78;
break;
case 70:
energy=67;
break;
case 60:
energy=56;
break;
//additional values go here
}

The keyword switch is used to start a case statement and is followed by the variable to be checked inside parentheses. Each
condition is introduced by the keyword case. The body of the switch statement is enclosed in braces.
At the end of each case in a switch statement, you will generally use the statement break to finish that case.
If you omit the break statement at the end of a case, execution will continue into the next case. Although this can sometimes be
useful, it is something to be aware of.
Another keyword that can be used is default. This performs a task for any other condition not met by the case statement. Taking
our above example, we could make our player's energy decrease more realistic by doing the following:

switch(health)
{
case 100:
energy=100;
break;
case 90:
energy=89;
break;
case 80:
energy=78;
break;
//additional values go here
default:
energy-=0.5;
}

This means now that every time this case statement is executed, it will set our energy to a predetermined value every time our
health reaches a number divisible by 10, and just subtracts 0.5 off in all other circumstances.
Notice that even a default case still has a break. You can put the cases in any order you wish.

The expression inside the switch condition can be any expression that results in a value, so long as the result is numeric. For
example, you could have:

switch(number % 100)

However, each case inside the switch, must be a constant: you cannot put expressions such as:

case n < 5:

or

case number:

If you do, the compiler will signal an error.

Also, you cannot have more than one case with the same value, or more than one default case.

As always, switch...case statements can be nested if necessary.

8. Loops

After having experimented a bit with if statements it is only fitting that you should be introduced to the concept of loops, as they
are fairly similar. A loop is a section of code that is executed over and over again until a certain condition is true. They are used
extensively in games, for example you will want a loop that continuously checks for certain keystrokes and whether some
particular events have occured and so on.
There are three ways in which to perform a loop, each of which we will discuss in this chapter.

8.1. While loops

Take a look at the following code example.

while(key_pressed(KEY_ESCAPE)==false)
{
wait(5);
}

Confusing? No problem, we will go through it bit by bit. The first line is really the most complicated one. First you have the word
"while", which means that a loop should begin here. After that you have the condition that needs to be met for the loop to RUN.
In this case, we use the function called key_pressed to check if the escape key has been pressed, and make the loop run while
that function reports that the key is not in fact being pressed. In short, the loop will continue to run until the user presses escape.
The code between the while statement and closing brace is then executed over and over again, and after each execution the
condition is tested once again to see if it is now true. If it is, the code after the closing brace will begin executing instead. In our
loop we only make the program wait for 5 milliseconds, which basically pauses execution for a very short time between every
check of our condition. If we didn't include this line, the processor's CPU would begin running at 100 % which is not really what
we want. So in any loop that runs for a longer period of time you should always have a short pause, and 5 milliseconds is usually
a good compromise between speed and CPU usage.

But let's say that you wanted to perform a particular task x number of times, would a loop be the right option? Certainly. Here is
a quick example that shows how you can make something happen 1000 times.

int x=0;
while(x<1000)
{
// Put your code here.
x++;
}

We start by making a variable called x and giving it a value of 0. Then we begin our loop, and tell it to execute while x is lower
than 1000. After this we insert any code that we want to execute, and finally we increase x by 1. This way, x will be incremented
by 1 every loop cycle so that it finally has the value 1000. When it does, the next check of our condition will obviously be false
since x is no longer lower than 1000. Effectively, we have then executed our code 1000 times with x going from 0 to 999. We
could of course have given x an initial value of 1 and let the loop run while it was lower than 1001, or by all means while it was
lower than or equal to 1000 (see the chapter on if statements for details), but there is a particular reason why we chose to start at
0 which will be explained further in the next chapter.
Here is another example. We will display the numbers 0 to 10 in a message box.
int x=0;
string numbers="";
while(x<=10)
{
numbers+=x;
if(x<10)
{
numbers+=" ";
}
x++;
}
alert("Printing numbers 0 to 10", numbers);

The loop executes while x is less or equal to 10, in other words, until x is 11.

We could slightly shorten this code, as follows:

int x=0;

string numbers="";
while(x<=10)
{
numbers+=x++;
if(x<=10)
{
numbers+=" ";
}
}
alert("Printing numbers 0 to 10", numbers);

Here, we have changed x in exactly the same place as we checked it. X will change after its value has been checked (notice the
++ operator on x).
We also slightly changed our if condition to check if x is less than or equal to 10. This is because otherwise the spaces would
stop from 9 upwards, since x is being increased before the if condition is checked.

8.2. Do while loops

We have looked at while loops, now it's time to show you something very similar. Let's start off with waiting for the user to press
escape again:

do
{
wait(5);
}
while(!key_pressed(KEY_ESCAPE));

There are two points of interest here. First, the line that says do tells BGT where the loop starts. The block of code, again
enclosed in braces, is the section of code that is to repeat while the condition is true. Finally, there is one additional line after the
close brace which tells the interpreter which conditions should be met in order for the loop to continue.
You may notice that in this while statement, there was an exclamation mark before the condition. This tells it to repeat the loop as
long as the condition is false. In short, the following two lines mean exactly the same thing:

while(key_pressed(KEY_ESCAPE)==false)

while(!key_pressed(KEY_ESCAPE))

It is important to remember that after the while statement of a do while loop, there is a semicolon, since this is the equivalent of
giving an instruction. Do the above, while this condition is met.
There is one notable difference between a while loop and a do while loop. A while loop checks the condition first, meaning that
the code could effectively be totally skipped if the condition is not met. However with a do while loop, the condition is not
specified until the end of the loop, which means the code is always executed at least once.

8.3. For loops

The for loop is the most powerful of the loops, and is used mainly for counting. Essentially, it combines the various parts of loop
control into one statement.
Consider the while loop in the above examples. Before we started the loop, the variable was initialised to a starting value (in our
examples, it was 0). Then, we had the while condition at the top of the while loop. Finally, just before the closing brace of the
while, we re-initialised x ready for next time around.
In a for loop, these three stages are packaged into one line:

for(int x=0; x<=10; x++)
{
//loop code goes here.
}

There are three sections in a for statement:
First, we initialise a variable to a starting value. Here, we chose 0. Although we created the variable from within the loop, it is
perfectly legal to use an existing variable, as if you were modifying its value anywhere from the script.
Second, we tell it upon what conditions our loop will continue to run. In our case, if x is less than 10.
Lastly, we tell it how to reassign the variable ready for the next cycle of the loop. Although we increased x by 1 as would be the
case in most other languages, the reassignment expression can be anything you like. Take a look at the following code.

for(x=1; x<1000; x*=2)
{
//code goes here
}

Every cycle of the loop will now multiply the current value of x by 2.

8.4. Break and Continue

There are two last possible scenarios that we need to cover. Sometimes you will want a lot of things to occur continuously, but
you may not know exactly when you want the loop to stop or you may want to stop it for several different reasons that can't all
be used as the condition. In this case, you might do something like the following:

while(true)
{
if(something==true)
{
break;
}
if(something_else==true)
{
break;
}
if(yet_another_variable==true)
{
continue;
}
// more code here...
wait(5);
}

There are three new things here. The first thing is on line 1, where we don't exactly have a condition for our loop. All we say is
"while(true)", which doesn't really make a lot of sense at first glance. However, it's not so strange because true, of course, is
always true which means that the loop will never end simply because the condition can never be false. So in short, we have
created an endless loop. Naturally we don't want the loop to go on forever; we do want to stop it at some point, but we do it
from inside the loop itself. The above three if statements check for conditions that don't actually exist, so if you ran this script it
would give you an error. What I am trying to show is that you may, at your own discression, break out of the loop for any reason
and at any time using the break statement that you see inside the first two if's. We briefly saw the break statement in action when
looking at the switch...case statements. The break statement basically forces the loop to terminate and begins executing the code
that comes after it. As mentioned before this is very useful when you want to exit the loop for a number of reasons, or when you
don't know exactly when it should stop.
It must be noted that, if the loops are nested, the break statement will only break out of the loop it is declared in.

The other keyword you saw in the third if statement is continue. Again, this keyword can be used anywhere from inside a loop.
The continue keyword tells the interpreter to skip the remainder of that cycle and move onto the next. This is useful, for example,
when you are reading from a file and there are certain things, like blank lines that need to be skipped.

9. Arrays and dictionaries

So far, we have seen the basic uses of variables. We have seen how they can be manipulated, and how they are used in a
general programming context. There is, however, a great issue that you will undoubtedly discover sooner or later, and that's the
incredibly large amount of different variables that you would have to keep track of even in a middle-sized game. Imagine if you
will, that you have a game board with 50 squares on it. The user is supposed to roll a pair of dice, their position increases by the
resulting amount, and then you want to check what should happen on the square that the player lands on. For the purposes of
this simple example we will assume that the variable representing a square can have 3 values; 1, 2, and 3. These values can mean
anything that you want, of course. 1 could mean that the user goes back three squares, 2 could mean that they get to remain
where they've landed and 3 could mean that they get to move forward three squares. Now let us try and construct part of this
game board quickly. We will let the first square be number 0 rather than number 1, for reasons which will become clear shortly.

int board0=2;
int board1=2;
int board2=2;
int board3=2;
int board4=1;
int board5=2;
int board6=2;
int board7=2;
int board8=2;
int board9=3;
int board10=2;
int board11=2;
int board12=2;
int board13=2;
int board14=1;
...

Let us stop there, as you will undoubtedly be furious at the amount of paper that you will have wasted if you printed out this
tutorial. You will, after looking at this example, agree that the code is incredibly repetitive and tedious? Let's say instead that this
was a side scrolling game and the different numbers represented ground, fire and water and the like, you might have had 300
squares instead if the level was a large one. Imagine the joy of writing out 300 such lines as you saw above... It's just not
practical. Not only does it take ages to write the actual contents of the grid or board, but you would have to have 300 if
statements in order to establish which of the 300 variables should be examined after the player moved to a new square. Can you
say nightmare? I thought so. Luckily, there is an easy enough solution... Arrays.

An array is, simply put, a list of variables. The array itself has a name like any other variable, but it contains different values that
you can refer to by index. An array can be as small or as large as you need it to be. As always, we will look at an example to
wet our apetite.

int[] board(50);
for(int x=0; x<50; x++)
{
board[x]=2;
}

Wait, wait... Many new things here. Let's go through the code line by line as usual. The first line tells the engine to create a new
array, which is to say a list of variables, of type int. The square brackets following int tells the engine that this variable is an array.
The array is then assigned to the variable called board, so that we have a way of referring to entries inside the array later. The
number inside the parentheses tells the engine to give this array 50 entries. On the next line we use a for loop to make a new
variable called x, and then start a loop that will run while x is lower than 50. On the line after that, something very interesting
happens. Here we assign a value of two to an entry in the board array. This entry is not specified with a literal value in this case,
however. It could certainly have been, but it makes much more sense to use a loop since we then do not need to specify each
individual entry explissitly as, of course, we would then be nearly as badly off as with the individual variables in the previous

example. You will observe that the index of the array, which is to say the entry in the list to access, is specified between a left and
a right bracket and not with a left and a right parenthesis, which you might have expected. On the last line we simply tell the
engine that our loop ends here. What we have done is to go through the entire list of 50 items, automatically assigning a value of
2 to each one. After this we only have to assign values manually to those entries in the list that are supposed to be something
other than 2. Quite a bit simpler than the nightmare example from earlier, right? I thought so.

Now you may also have understood the reason for our specifying the first square on the board in the original example as 0 rather
than 1. This is simply because the first index in an array is always 0, never 1. So if you have an array with 100 entries and you
wish to loop through it to perform some task, you would start at 0 and go up to 99. If you try to access an entry in an array
which is out of range, you will get a runtime error. This means that the interpreter will not complain until the array is evaluated,
therefore it is absolutely essential that you test the game at regular intervals, especially any array related sections, otherwise you
may find you receive some rather angry players on your back.

You can treat an array entry just like a regular variable, which means that you can use it in if statements and loops and anything
else that you can do with normal variables, with the only difference that you specify an index in the array rather than a unique
variable name.

You may also have arrays with multiple dimensions, in order to make things such as an x-y grid or even x-y-z. For example, if
you want to make a chess board you could do the following:

int[][] chessboard;

Multidimensional arrays are, to put it simply, arrays of arrays. There is a current limitation in the array usage that prevents you
from specifying the size of each dimension in a multidimensional array. To do this with our chess board, for instance, we must
manually resize the first dimension and then use a loop to also resize each entry in the second dimension, as follows:

chessboard.resize(8);
for(int i=0; i<8; i++)
{
chessboard[i].resize(8);
}

At this point, we have an array called chessboard which is 8 by 8 in size. Please note that you cannot put this resizing code in the
global scope, it must be in a function such as main where as the actual declaration of the chess board may still be global.

To access an element in an array with multiple dimensions you use the same exact method as when you accessed elements
before, just with the different dimensions specified between brackets. To access the square in the upper left corner of our chess
board, for instance, one could write:

chessboard[0][0]=5;

And to access the square in the lower left corner, you would do the following:

chessboard[0][7]=5;

The same goes for a board with three dimensions. If you wanted a board which was 3 by 3 by 3, you would declare and initialize
it like this:

int[][][] board;
board.resize(3);
for(int i1=0; i1<3; i1++)
{
board[i1].resize(3);
for(int i2=0; i2<3; i2++)
{
board[i1][i2].resize(3);

}
}

You could then access an element by saying:

board[2][0][1]=10;

By using an array with three dimensions you could, for example, represent a grid with x, y and z coordinates. This would be a
true 3d game with left, right, back, forward, up, and down movement.

At this time, the BGT engine supports arrays with up to 4 dimensions.

Arrays are used for many different things. As a matter of fact, a string itself is an array, though it doesn't look it on the outside.
Each element of a string array, is a single character.
Here is an example:

string my_string="This is a string.";
alert("my_string","my_string entry number 2 is the letter "+my_string[2]);

This would return the letter i. Remember that arrays always start with 0, therefore entry 2 holds character 3.
As strings can be used as function parameters and return values, so can arrays. However there may be times when you don't
know your array's length, either because it is a returned string or array, or because a loop is always changing values depending
on certain conditions. No problem. An array holds two special functions to allow you to check or change your array. Length is
self explanatory. It returns the length of the array. Please note that this does not mean the final element number. If you are using
this method as the condition of a loop you need to check that your counter is less than that of length, as shown in the below
example, which will display every character of the string in a message box:

string my_string="string";
for(uint my_counter=0; my_counter < my_string.length(); my_counter++)
{
alert("my_string","my_string["+my_counter+"]="+my_string[my_counter]);
}

Notice that to obtain the length we say my_string.length(). This is because the function length() is stored within the variable. This
is known as a method. We will discuss methods in more detail in the next chapter. Also you will have noticed that I used a
variable of type uint rather than int. This is because the length method returns an uint as opposed to an int or a double. It is
always recommended to have signed and unsigned integers matching, otherwise the compiler will flag a warning when you
attempt to run your script.
There is a final method that is used to control an array. it is called resize, and is used to, you guessed it, resize an array, either
adding new elements or removing elements. It takes one parameter, the number of elements that the new array should contain.
Note that, like length, you are declaring the actual number of elements that should be present, not the number of the last element.
For example:

string my_string="this is a string.";
my_string.resize(4);

This will resize the array down to 4 elements, ending on entry 3. Note that when you resize an array to smaller than its original
value, entries are removed from right to left, I.E. from the last element backwards. Therefore the resulting string will be "this".

It is possible to specify exactly which values will go into an array at the point where the array is defined. To do so, simply put an
equals sign after the name of the array, and follow it with a list of values in braces. It sounds more complicated than it is. For
example, let us define an array of integers and initialize it with some meaningful values, all in the same line of code:

uint[] primes = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };

There is a slight limitation with arrays, and this is that they can only store values of the same data type. You could not, for

example, have one entry with the value 45, and another with the value "hello". However, this can be resolved with another type
of variable, known as dictionaries. A dictionary is what is called an object with methods, which I touched upon briefly in our last
example. More on this in the next chapter.

Dictionaries are another way to declare variables. However, unlike standard variables, you declare the variable name using a
string, and it can store any value. Therefore, you could have a dictionary holding several variables, all with a name and a value,
such as the boards we described in our examples above. Let us now use the same example, this time using dictionaries.

dictionary board;
for(int x=0; x<50; x++)
{
board.set(x, 2);
}

Again, lots of new things to discuss. The first line simply declares a dictionary, which we refer to as board. The next line then
starts a loop, like we did with the array, from 0 to 49. The line after that tells the interpreter to create an entry in the dictionary,
called the current value of x, and assign the number 2 to that entry.
This has certain advantages over arrays, since you could in theory store values of x and y coordinates, like so:

dictionary board;
for(int x=0; x < 4; x++)
{
for(int y=0; y < 4; y++)
{
board.set(x+","+y, 2);
}
}

We accomplish this by using a nested loop to increase both counters and assign the initial value of 2 to each square. Notice that
we are using two values separated by a comma inside our name string. This just makes it easier for us to separate the squares
from one another, as the dictionary object just sees it as a string; it has no idea what the name might mean to us. In practice, all
that is happening is that a string with a unique name is assembled which is to say our square numbers separated by comma.

Similarly, to retrieve the value of a certain square, you may do the following:

int square_value;
board.get("2,2", square_value);

The second parameter of the get method is passed by reference, meaning the value is stored in the variable that we pass it. If my
meaning is unclear, refer back to the functions section which contains detailed information about the use of parameters.

As well as being able to set and retrieve values in dictionaries, you can also change, and even delete them:

if(board.exists("2,2"))
{
board.delete("2,2");
}

This tells the interpreter to look in the dictionary called board for "2,2". If it finds it, this value is then deleted. This cannot be
done with standard variables.

It is very important that you check the existence of a dictionary entry. If you delete or check an entry that doesn't exist, this will
not cause any syntax or runtime errors to occur, but the variable you are attempting to use to retrieve the value will not change.

You can also use the delete_all method to delete all assignments from the dictionary.

Please note that dictionaries are significantly slower than arrays and regular variables, so use them with care.

10. Objects and classes

Objects are a very powerful and absolutely essential feature when making games with the BGT engine, so be sure to read this
chapter very carefully. An object is a regular variable, but with special functionality. You can think of an object as a variable that
contains other variables inside itself, as well as functions that do something with that particular object. Another name for an
object is a class. To make things a little more clear, we will begin with an example.

sound ambience;
ambience.load("curry.ogg");

This example does several things. It tells the engine that we want a new object of type sound, that we want the object to be
assigned to a variable that we call ambience, and finally it calls a function within the ambience variable, specifying the filename of
the sound to open.

When an object is created, it can take a list of parameters which largely determine how the object will behave. It is also possible
for objects to take no parameters at all, which the sound and timer are good examples of. The timer object would be created as
follows:

timer jumptime;

Once an object has been created, you may begin using it. An object has what is called properties, which are basically variables
that are stored inside the object and which are used both to get and set various pieces of information during the lifetime of the
object. The sound object, for instance, has a property called volume. This determines how loudly the sound will play, and can be
modified in real-time. Here is an example of how to create a sound object with a sound that is opened as a stream, and then how
to set its volume to -6 decibel which which is to say about half of the original amplitude:

sound ambience;
ambience.stream("curry.wav");
ambience.volume=-6;

Pretty simple, right? It is of course also possible to check the value of a property in our object, like this:

if(ambience.volume>-10)
{
// Do something
}

In this case, we do something if the volume is greater than -10 decibel. Some properties cannot be modified but only checked,
while others can only be modified at certain times but not at others. You will find all the information about this in the properties
list for each object in the object reference.

Objects also have what is called methods. A method is simply a function like any other that you will find in the BGT engine, with
the only difference that it works exclusively with a particular object. To go back to the sound object, there is a method called
play_wait which starts the sound playback and then waits for it to finish before returning. In fact, the load and stream functions
we experimented with earlier are also good examples of methods.
Let us extend the previous example to include playing the sound as well as setting its volume.

sound ambience;
ambience.stream("curry.wav");
ambience.volume=-6;
ambience.play_wait();

Familiar? Good, as we are basically just calling a function but one that operates on our specific copy of the sound object, called

ambience in this case. play_wait takes no parameters, but many other object methods do in which case they are specified
between a left and a right parenthesis just like in a regular function call.

It is perfectly possible to make two versions of the same object, in which case they will work completely independently of each
other. If you make two sound objects for example and then call play on both, you will get both sounds playing at the same time.

You may want to pass objects around various functions of your own. To do this, it is essential that you pass handles, and not the
object itself. A handle is essentially a reference to the original object. To do this, you would do something like the following:

sound@ play_sound(string filename)
{
sound the_sound;
the_sound.load(filename);
the_sound.play();
return the_sound;
}
void main()
{
show_game_window("My Game");
sound@ ambience=play_sound("curry.wav");
ambience.volume=-6;
@ambience=null;
}

In short, when you declare an object handle, you put an @ (at) sign after the variable type. You can then use this variable as if it
were a standard object, the only difference being that you are referencing the original variable. When you want to change the
value of your handle, for example to make it point to another object of the same type, you simply put the @ (at) sign before the
variable name.
You can destroy a handle by setting its value to null. This is the object equivalent to 0. With this in mind, though you cannot
manually destroy the contents of an object, a way to work around this is to set all your global objects as handles, and only use
the objects themselves in functions. This way, although the objects themselves are local variables, they have global variables still
referring to them. Therefore the object will always remain usable until its last handle is destroyed.

You can also make your own objects, also known as classes, with their own methods and properties. For example, you could
make a class for an enemy, as follows:

class enemy
{
int health;
int speed;
int position;
void fire_weapon()
{
//weapon code goes here
}
void move(int direction)
{
//movement code goes here
}
}

The only new line here is the first one, but it is only the same as declaring a variable, with the slight difference that you have to tell
the engine how to make that class. In this case, our enemy class has three properties, health, speed and position, and two
methods, fire_weapon and move. However, we need not change the position property directly, since the move function will do
that for you, along with any other code that you put in there. It is important for this reason that you document which methods and
properties should or should not be touched.

To use our class we would then write:

enemy robot;

Then our class would be ready for use, just like the objects provided in BGT.

robot.move(right); //assuming that right is a constant
robot.fire_weapon(); //makes the robot fire his weapon

There is a slight problem though. The enemy's health hasn't been assigned a value. This means that every time we declared an
enemy variable, we would have to set every health to 100. This is easily solved with the constructor and destructor functions.
The constructor function simply instructs the engine how our properties should be set up, and/or performs any other tasks that
we might want to have done whenever a new instance of our class is created. The constructor function is optional, and is
declared with the same name as the class. The destructor function is called when our class instance is destroyed. It can be used
to do any necessary clean-up work. The destructor is also optional, and is declared in a similar fashion as the constructor but
with a ~ (tilde) sign prepending the class name. Finally, both the constructor and destructor functions are the only functions which
do not declare a return type; not even void.
Let us extend our enemy class to include a constructor:

class enemy
{
int health;
int speed;
int position;
enemy()
{
health=100;
position=0;
speed=300;
}
void fire_weapon()
{
//weapon code goes here
}
void move(int direction)
{
//movement code goes here
}
}

Now, every time we declare a variable of type enemy, its properties will always be set correctly.

Note that it is also possible to write the following:

class enemy
{
int health=100;
int speed=300;
int position=0;
void fire_weapon()
{
//weapon code goes here
}
void move(int direction)
{
//movement code goes here

}
}

In this case, we don't actually implement a constructor because the properties are given default values directly in their declaration.
However, the constructor may need to do other work so we will stick to using it in the future as well.

To make your class more portable you can use parameters for your constructor, rather than expecting the user of the class to
modify the properties directly. We show this with our rapidly growing enemy class:

class enemy
{
int health;
int speed;
int position;
enemy(int init_health, int init_speed, int init_pos)
{
health=init_health;
position=init_pos;
speed=init_speed;
}
void fire_weapon()
{
//weapon code goes here
}
void move(int direction)
{
//movement code goes here
}
}

Now to declare our enemy, we would write the following:

enemy robot(100, 300, 0);

This would set the properties of our enemy object instance to the values that we specified. That way you can create enemies of
varying strengths, speeds and randomly scattered on the game board. Note that since we define only one constructor which
takes parameters, it is no longer possible to declare our enemy without specifying these parameters (see below for more
information on automatic default constructors).

As I mentioned in the functions section, functions can have the same name, so long as the parameters are different. This also
holds true for class methods, including the constructor, but not the destructor. This means, that if you wanted to allow your
enemy to be set with different values, you could have a constructor where the user must specify the parameters, but you could
also have a constructor where no parameters have to be given. This way, you can decide whether to declare your enemies with
parameters or not. The compiler will then find the constructor that matches your parameters (if any), and call it for you. We will
now demonstrate this with our enemy code. I will not include the class properties or methods, I will simply use our constructors.

class enemy
{
//properties go here
//constructor without parameters
enemy()
{
health=100;
position=0;
speed=300;
}

//constructor with parameters:
enemy(int init_health, int init_speed, int init_pos)
{
health=init_health;
position=init_pos;
speed=init_speed;
}
//class methods go here
}

Of course, you could put your class methods wherever you liked, however it made logical sense to put our constructor at the top
of our class, since this is the first function it calls.
Now to declare our enemy, since we have two constructors, we could either do

enemy robot;

or

enemy robot(200, 150, 0);

The first declaration would simply give us an enemy with his default values. However, our second declaration gives our enemy
twice the strength and speed of our first. Although 150 is half the value of 300, we are telling the program the time in which it
takes him to move, not how fast he moves in a given time.

if you do not define a constructor in your script, the compiler will automatically make an empty one for you that takes no
parameters. This means that by default, we could declare any of our classes as you saw above with no argument list. However, if
you define a constructor that takes parameters, the default one is no longer automatically implemented. It is only implemented
automatically by the script compiler if there is no constructor at all specified by you. Therefore, if you want a default constructor
(which is to say a constructor that takes no arguments), as well as a constructor that does take arguments, you must implement
both explicitly just like in the example above.

Please note that, as mentioned earlier in this tutorial, primitive variables (which is to say variables that are not objects), will
contain undefined values if you don't give them initial values explicitly. This applies not only to global variables and variables
inside functions and methods, but also to properties in classes. If you declare an object like:

enemy robot;

Then the default constructor (if present) will be invoked implicitly, so unlike with primitive variables such as int and float, the
contents of the robot variable is not at all undefined. The constructors are supposed to initialize the properties used internally by
the class, however. That does not happen automatically. In short, you must make sure that all the properties that you use in a
class are initialized to meaningful values before they are used - just like you do with primitive variables outside of classes. The
constructors are a good location in which to do this. Alternatively you can give the properties initial values when they are
declared, just like you would with global variables or local function variables.

Destructors more or less work in the same manner as constructors, except that they are called when your class is destroyed. This
is usually either when its last reference is destroyed, or when the program exits. Here is our basic enemy once more, with a single
constructor taking no arguments, and a destructor:

class enemy
{
//properties go here
enemy()
{
health=100;
position=0;
speed=300;

}
~enemy()
{
alert("glory!","your enemy is dead!");
}
//methods go here
}

It would be pointless to run this example. In fact, if you implemented this example into a game, once you exited the game you
would get message boxes left right and centre telling you your enemy was dead, because the interpreter is destroying all active
objects and therefore calling their destructor functions, if any are present. Just like with the constructor, the compiler will make an
empty destructor for you by default if you choose not to create one yourself. This means that you only need to define one
explicitly if you wanted to perform some specific action when the class was destroyed.

11. Advanced object techniques

The previous chapter has given you just enough information to begin using objects productively in your game creation endeavors.
This chapter will extend your arsenal of object-oriented techniques. Note that the concepts introduced here are considered
advanced, and you may well decide to skip this chapter on your first pass through this tutorial.

11.1. Objects in a nutshell

Before we delve into advanced concepts, let us briefly take stock of what you already know about objects. Here is a quick
rundown of object-oriented terminology. If anything appears unclear, you may wish to reread the previous chapter as it lays the
foundations for this one.

An object can be imagined as a box with variables and functions living inside it. The variables inside an object are called
properties, and the functions inside an object are called methods.

Every object belongs to a class, or type, and an object's class determines which properties and methods the object has. For
example, a sound object, belonging to the sound class, will always have a play method because the sound class says so.

Whenever an object is created, a special function called a constructor is executed. Likewise, whenever an object is destroyed, a
special function called a destructor is executed. Constructors and destructors are defined by the object's class. A class may
provide multiple constructors with different parameters. However, it is not allowed for a class to have more than one destructor.

11.2. Inheritance

Imagine you are creating a game in which the player character may meet different kinds of animals, and let us assume you have
written a class to represent birds in your game. The bird class might look something like the following:

class bird
{
void walk()
{
// Walking code for birds goes here.
}
void run()
{
// Running code for birds goes here.
}
void fly()
{
// Flying code for birds goes here.

}
}

// Let's create a few birds
void main()
{
bird sparrow;
bird dove;
bird eagle;
sparrow.run();
dove.walk();
eagle.fly(); // Takes us up where we belong :-)
}

Now, in an effort to make the game more realistic, we come up with the idea that the sparrow in the above code should actually
be able to sing. How might we go about this?

One approach might be to modify our bird class by adding a sing method. The obvious problem is that this would give every
bird, not just the sparrow, the ability to sing. Even in a game world, doves should still coo, and eagles should still cry. After all, it
was realism which prompted the addition of the sing method in the first place.

Our second approach would be to create an additional class for sparrows which is identical to the bird class in every way but
one: it has a sing method. Here is the corresponding code for both classes, in full:

class bird
{
void walk()
{
// Walking code for birds goes here.
}
void run()
{
// Running code for birds goes here.
}
void fly()
{
// Flying code for birds goes here.
}
}

class sparrow
{
void walk()
{
// Walking code for sparrows goes here.
}
void run()
{
// Running code for sparrows goes here.
}
void fly()
{
// Flying code for sparrows goes here.
}
void sing()

{
// Singing code for sparrows goes here.
}
}

This approach indeed solves our problem because now sparrows can sing but birds in general cannot. There are a few
disadvantages though:

1.
2.

3.

Our code has become lengthy and repetitive.
Our code has become harder to maintain. Imagine you would like to fix a bug in the fly method for all birds. In our running
example you would need to change two fly methods, one for the bird class, the other one for the sparrow class. It is easy
enough to fix the one but then to forget fixing the other. In such a scenario, even though you are sure you fixed the bug,
you will soon find out that it still persists. Even worse, your customers might be the ones to find out. Now assume you had
created classes for cooing doves and crying eagles in the same way. A bug in the fly method would then have to be fixed
in four places. Forget just one, and the bug will remain. Testing might even suggest it was fixed because the bug occurs for
some kinds of birds and not for others.
Suppose your game ended up containing an array of birds, like so:
bird[] aviary(50);
You would have no way of putting a sparrow into the aviary because all elements of an array must share a common data
type.

Wouldn't it be wonderful if we had a way to define a class for sparrows which inherits all the functionality from the bird class and
simply adds the sing method? And wouldn't it be even better if BGT somehow knew that a sparrow is just another kind of bird
so that we could place a sparrow into an array of birds? You will be delighted to know that such a way exists in BGT. It is called
class inheritance.

Suppose we have defined the bird class as above. Now we can define the sparrow class as follows:

class sparrow : bird
{
void sing()
{
// Singing code for sparrows goes here.
}
}

// Let's make some birds
void main()
{
bird b1;
bird b2;
sparrow s;
s.fly(); // Works because every sparrow is also a bird
s.sing(); // Works because s is a sparrow
bird@[] aviary = { b1, b2, s };
}

In the above code, the sparrow class inherits from the bird class. We call sparrow a derived class, or subclass, and we call bird
a base class, or superclass. These are all just different ways of expressing the same fact, and we take the liberty to use them
interchangeably because all of them are in common use.

To define a derived class, you follow the class name with a colon after which you write the name of the base class, as
demonstrated above. The derived class will inherit all properties and methods defined in the base class. It is as if every sparrow

object contained a hidden bird object. After all, part of being a sparrow is being a bird. In this zen-like statement, by the way, is
contained the entire concept of inheritance.

It is important to note that our aviary is an array of bird handles rather than an array of birds. When treating a sparrow as a bird,
it is vital that we use a handle. Consider the following code:

sparrow s; // A sparrow is created
bird@ b1 = s; // Through the handle b1, we can treat s as a bird
b1.fly(); // This works, and is equivalent to saying s.fly();
s.sing(); // This works because s is a sparrow
b1.sing(); // Does not work because, although b1 points to a sparrow, we are treating it as a bird
bird b2 = s; // Makes a new bird

That last line above is actually more complex than you might think. It creates a new bird b2 from the data contained in s, but b2
will not be a sparrow. You might say that b2 contains the birdness of s but not the sparrowness. This is quite different from b1
because b1 is not a copy of s but merely a handle through which s can be treated as a bird. b1 and s are names for the same
object in memory, but b2 is another bird altogether.

Not only can a derived class add methods to a base class but it can also modify the behavior of existing methods. Let's say we
wanted to give sparrows their own unique fly method. The sparrow class would then look like the following:

class sparrow : bird
{
void sing()
{
// Singing code for sparrows goes here.
}
void fly()
{
// Flying code for sparrows goes here.
}
}

void main()
{
bird b; // Create a bird
sparrow s; // Create a sparrow
bird@[] aviary = { b, s }; // Put handles to both of them into the same aviary
aviary[0].fly(); // Calls the fly method of the bird class
aviary[1].fly(); // Calls the fly method of the sparrow class
}

Here we have modified the behavior of a method by defining, in the derived class, a method of the same name and with the same
parameters (none in this case). We say that the fly method in the sparrow class overrides the one in the bird class.

Notice how inheritance provides a way to give equal treatment to objects of different types. To our aviary, the sparrow is just
another bird. This concept is called polymorphism. The word comes from Greek and means "of different form." In this case it
refers to the fact that we give equal treatment to objects which are different in form, or type, or class. In our running example,
polymorphism is possible because our two classes, bird and sparrow, are related by inheritance. Of course we could not place
any arbitrary bird into an array of sparrows for the simple reason that not every bird is a sparrow. On the contrary, we could
neither store a sound object in an array of birds, nor could we store a bird object in an array of sounds, because bird and sound
are unrelated.

While properties and methods are inherited, constructors are not. However, since every sparrow object contains a hidden bird

object, two constructors need to be executed when a sparrow is created. Just as a house is built from the foundations upwards,
it is the base class constructor which executes first. The opposite applies when an object is destroyed. In this case, the
destructors are called from the derived class upwards, so the sparrow destructor will come first. You can memorize this by
reminding yourself that destroying something is the opposite of creating it.

Another way of looking at it is to imagine that the first thing a derived class constructor does is to call a parameterless base class
constructor. This happens automatically, and in most cases is just what you want. There are cases, however, when you would
rather call another constructor in the base class. For this scenario, BGT has a special keyword called super. You can use the
super keyword just like the name of a function, only it refers to a constructor of the base class.

Suppose the bird class had an additional constructor which would take the bird's wingspan as parameter:

class bird
{
double wingspan;
bird(double wingspan)
{
this.wingspan = wingspan;
}
void walk()
{
// Walking code for birds goes here.
}
void run()
{
// Running code for birds goes here.
}
void fly()
{
// Flying code for birds goes here.
}
}

The above code uses the "this" keyword which you may not have seen before. Anywhere within the body of a class definition,
the keyword "this" can be used to refer to the object currently considered. In case of a constructor, this would of course be the
object currently created. In the above case the keyword is necessary because there are two things to which the name wingspan
could possibly refer to, one of them a property of the object to create, the other a constructor parameter. If we had merely
written
wingspan = wingspan;
then we would have assigned the constructor parameter called wingspan to its own value, a valid but pointless operation.

Now we can modify the sparrow class so that it calls our new constructor:

class sparrow : bird
{
sparrow()
{
super(3); // Call base constructor with parameter of 3
}
void sing()
{
// Singing code for sparrows goes here.
}
}

In this way we have specified that a sparrow is always created with a wingspan of 3. The elegance of this approach is that we
did not have to touch the wingspan property directly but could instead use code which was present in the base class. As you may
have realized by now, code reuse is what object-oriented programming is all about.

Let us add classes for eagles and doves just to demonstrate how easy it is. And while we are at it, we will give each bird a
make_sound method which causes it to produce its characteristic sound. Here is our entire code, in full:

class bird
{
double wingspan;
bird(double wingspan)
{
this.wingspan = wingspan;
}
void walk()
{
// Walking code for birds goes here.
}
void run()
{
// Running code for birds goes here.
}
void fly()
{
// Flying code for birds goes here.
}
void make_sound()
{
// Code for generic bird sound goes here.
}
}

class sparrow : bird
{
sparrow()
{
super(3);
}
void sing()
{
alert("How beautiful!", "The sparrow sings a little melody.");
}
void make_sound()
{
sing(); // Could have used the "this" keyword but didn't have to
}
}

class eagle : bird
{
eagle()
{
super(20);
}
void cry()

{
alert("How atmospheric!", "The last eagle cries over the last crumbling mountain.");
}
void make_sound()
{
cry();
}
}

class dove : bird
{
dove()
{
super(8);
}
void coo()
{
alert("How unnerving!", "A dove coos its love song in a language only doves find lovely.");
}
void make_sound()
{
coo();
}
}

// Let's make some birds
void main()
{
sparrow s;
eagle e;
dove d;
bird@[] aviary = { s, e, d };
for(uint i=0; i<aviary.length(); i++)
{
aviary[i].fly();
aviary[i].make_sound(); // Each according to its kind
}
}

You might wish to run this code to watch polymorphism in action. Notice how our for loop just tells every bird to make its
characteristic sound, and each bird behaves quite differently. If later on we added another kind of bird to our aviary, the for loop
could remain unchanged. This leads to the fascinating observation that old code can call new code.

A tool is only really powerful in the hands of a person who knows not just how, but also when to use it. In the case of
inheritance, a good approach is the following: Use inheritance when not using it would require you to maintain several identical
copies, or similar variations, of the same code. Use inheritance when modelling real-world concepts which are themselves
hierarchical. Finally, use inheritance when objects of very similar but not absolutely identical types will be given equal treatment.
With some practice you will recognize those situations early on in the design phase before you have even written the first line of
code. The art and science of what this paragraph talks about is called object-oriented design, just in case you would like to do
some research on it.

A tool is even more powerful in the hands of a person who, in addition to knowing when to use it, also knows when not to use it.
So we close this section with a word of warning about inheritance. While it is certainly a powerful technique which has many
valid uses, many programmers, when first learning about it, get carried away by its elegance and conclude that finding just the
right class hierarchy must be the solution to every possible problem of software design. If you aren't careful, your class

hierarchies may increase beyond the reasonable limits because you would like them to cover just about every subclass
imaginable. For example, we might have defined separate classes for every kind of eagle or dove known to science. BGT will
uncomplainingly let us extend our class hierarchies to arbitrary unfathomable depths. But unless our game will be about
ornithology (the branch of zoology that studies birds), the two-level hierarchy presented above will probably be more than
enough. If the truth be told, in most cases even a single bird class should suffice. Or to quote Python inventor Guido van Rossum:
"Simple is better than complex. Complex is better than complicated."

11.3.  Interfaces

In the previous section you learned about the powerful concept called polymorphism. One way to achieve polymorphism is
through inheritance because BGT allows us to refer to an object via a handle to its base class. Inheritance makes sense when
classes share a lot of code or lots of conceptual similarities. Sparrow and dove, for example, share all the code for walking,
flying, and running, and in addition they are conceptually similar due to the fact that they are both birds.

It is possible, however, to conceive of scenarios in which objects share neither a lot of code nor lots of conceptual characteristics
but still might be given equal treatment. Consider the concept of a sound source. Looking at the last code example of the
previous section, you will find that a bird may be called a sound source. After all, it has a make_sound method. Now let us invent
another kind of sound source, one which is almost entirely unlike any bird.

class musical_instrument
{
uint complexity;
musical_instrument(uint complexity)
{
this.complexity = complexity;
}
void make_sound()
{
alert("Info", "You hear the sound of music.");
}
}

class drum : musical_instrument
{
drum()
{
super(2);
}
void make_sound()
{
alert("Info", "You hear the steady beating of a drum.");
}
}

Birds and musical instruments are different in every way but one---they are both sound sources. But this similarity alone warants
the idea of giving them equal treatment, for example by storing handles to them in an array of sound sources.

One way of achieving this would be to implement bird and musical_instrument as derived classes with a common base class
which we call sound_source. In the previous section we formulated some criteria for when inheritance might be useful. Let us see
if they apply to birds and musical instruments:

Use inheritance when not using it would require you to maintain several identical copies, or similar variations, of the same code.
This is certainly not the case because bird and musical_instrument share no code whatsoever except a common method name.

Use inheritance when modelling real-world concepts which are themselves hierarchical. This is clearly a borderline case. One

might argue that since both a bird and a musical instrument are sound sources, there is some form of hierarchy between the
concepts. In practice, however, the concept of a sound source is so abstract in relation to the concepts of musical instruments
and birds that we do not perceive this hierarchy in our everyday thinking.

Finally, use inheritance when objects of very similar but not absolutely identical types will be given equal treatment. Note the
expression "very similar." Birds and musical instruments are not even considered remotely similar in everyday discourse.

Try as we might, it seems we cannot create a good case in favor of inheritance. But there is another way of expressing similarity
in one respect for types which are dissimilar in every other respect.

A bird and a musical_instrument share a common method signature. By the word signature we refer to a combination of name
and parameters, so what we mean is that both classes have a method of the same name and with the same parameters, none in
this case. Another way of putting this is stating that bird and musical_instrument implement the same interface consisting of a
parameterless method called make_sound. Here is the code which defines this interface:

interface sound_source
{
void make_sound();
}

Note the semicolon that immediately follows the parentheses after make_sound. In an interface definition we do not include the
bodies of methods. An interface is merely a list of method signatures, and if a class contains methods with the given signatures of
an interface, we say that the class implements that interface. The whole point is that an object can be manipulated through a
handle to one of its interfaces.

We have to tell BGT that the bird class and the musical_instrument class both implement the sound_source interface. The syntax
for this is exactly the same as for inheritance, so instead of
class bird
we write
class bird : sound_source
and instead of
class musical_instrument
we write
class musical_instrument : sound_source

Here is a code example for everything we have learned about inheritance and interfaces:

interface sound_source
{
void make_sound();
}
class bird : sound_source
{
double wingspan;
bird(double wingspan)
{
this.wingspan = wingspan;
}
void walk()
{
// Walking code for birds goes here.
}
void run()
{
// Running code for birds goes here.

}
void fly()
{
// Flying code for birds goes here.
}
void make_sound()
{
// Code for generic bird sound goes here.
}
}

class sparrow : bird // Note that this automatically implements sound_source
{
sparrow()
{
super(3);
}
void sing()
{
alert("How beautiful!", "The sparrow sings a little melody.");
}
void make_sound()
{
sing(); // Could have used the "this" keyword but didn't have to
}
}

class eagle : bird
{
eagle()
{
super(20);
}
void cry()
{
alert("How atmospheric!", "The last eagle cries over the last crumbling mountain.");
}
void make_sound()
{
cry();
}
}

class dove : bird
{
dove()
{
super(8);
}
void coo()
{
alert("How unnerving!", "A dove coos its love song in a language only doves find lovely.");
}
void make_sound()
{
coo();

}
}

class musical_instrument : sound_source
{
uint complexity;
musical_instrument(uint complexity)
{
this.complexity = complexity;
}
void make_sound()
{
alert("Info", "You hear the sound of music.");
}
}

class drum : musical_instrument
{
drum()
{
super(2);
}
void make_sound()
{
alert("Info", "You hear the steady beating of a drum.");
}
}

void main()
{
sparrow s;
eagle e;
dove d;
bird@[] aviary = { s, e, d };
for(uint i=0; i<aviary.length(); i++)
{
aviary[i].fly();
aviary[i].make_sound(); // Each according to its kind
}
drum my_drum;
sound_source@[] my_sound_sources = { d, s, my_drum };
for(uint j=0; j<my_sound_sources.length(); j++)
{
my_sound_sources[j].make_sound();
}
}

Note that, while a class may inherit from at most one other class directly, it may implement any number of interfaces. If you wish
to express that a class called c implements the three interfaces i1, i2, and i3, you simply separate the interface names by commas,
like so:

class c : i1, i2, i3
{
}

Interfaces and inheritance are not mutually exclusive, so a class may implement certain interfaces as well as inheriting from
another class. So to express that class c1 inherits from class c2 as well as implementing interfaces i1, i2, and i3, you would write:

class c1 : c2, i1, i2, i3
{
}

11.4. Operator overloading

This section, although having a complex-sounding title, is in fact much simpler than the two previous sections about inheritance
and interfaces. Operator overloading is what programmers call "syntactic sugar." It sweetens up your life and leads to very
readable and elegant code if used correctly, but you can do entirely without it if you so desire. It is, in other words, an optional
convenience feature.

Let us write a simple class for representing three-dimensional vectors. In case you need a refresher, a three-dimensional vector is
simply a combination of three numbers. For example, (3, 4, 5) is a three-dimensional vector. We say that a vector is composed
of three numbers, and the three numbers are the components of the vector. By convention, when dealing with a three-
dimensional vector, we call its first component the x component, its second the y component, and its third the z component. So
(3, 4, 5) has, for instance, a y component of 4. In mathematics, vectors are commonly used to describe locations in space. For
example, (3, 4, 5) can be interpreted as follows: From a fixed location called the origin, go three meters to the right, four meters
forward, and then fly five meters upward. And don't you start arguing with a mathematician that gravity won't let you---after all,
this is not physics class. Then again, I shouldn't have brought meters into this. In games, by the way, vectors can play an
important role because they are very adequate for describing where things are, how fast they move, and how quickly they
accelerate. They can even describe which way something or someone is facing. So never underestimate a vector even though it's
just three numbers.

Vectors can be added, and the way to add them is to add the individual components. For example, adding (3, 4, 5) to (5, 4, 3)
gives us (8, 8, 8).

A vector can be multiplied by a number called a scalar, and the way to do that is to multiply the individual components by the
scalar. For example, multiplying (3, 4, 5) by 6 gives us (18, 24, 30).

Finally, vectors can be compared for equality. The rules say that two vectors are equal when their individual components are
equal such that the two vectors share equal x, y, and z components. For example, (3, 4, 5) is equal to (3, 4, 5) but not to (4, 5,
3).

Let us translate into code what we know about vectors:

class test_vector
{
double x;
double y;
double z;
test_vector(double x, double y, double z)
{
this.x = x;
this.y = y;
this.z = z;
}
double get_x()
{
return x;
}
double get_y()
{
return y;

}
double get_z()
{
return z;
}
string to_string()
{
return "(" + x + ", " + y + ", " + z + ")";
}
test_vector add_to(test_vector@ other)
{
return test_vector(x+other.x, y+other.y, z+other.z);
}
test_vector multiply_by_scalar(double scalar)
{
return test_vector(x*scalar, y*scalar, z*scalar);
}
bool is_equal_to(test_vector@ other)
{
return x==other.x && y==other.y && z==other.z;
}
}

void main()
{
test_vector v1(3, 4, 5);
test_vector v2(5, 4, 3);
test_vector@ v3 = v1.add_to(v2);
alert("For your information", v1.to_string() + " + " + v2.to_string() + " = " + v3.to_string());
test_vector@ v4 = v1.multiply_by_scalar(6);
alert("For your information", "6 * " + v1.to_string() + " = " + v4.to_string());
if(v1.is_equal_to(v2))
{
alert("For your information", v1.to_string() + " and " + v2.to_string() + " are equal.");
}
else
{
alert("For your information", v1.to_string() + " and " + v2.to_string() + " are not equal.");
}
}

Observe the clumsy expression v1.add_to(v2) which we used to add two vectors. A real timesaver would be a possibility to
define addition of vectors in such a way that we could simply write v1+v2 and have BGT automatically do the right thing. This is
indeed possible and is called operator overloading. An operator is a symbol which stands for an operation. For example, the plus
symbol "+" stands for addition. Overloading is the process of extending the meaning of a symbol. For example, the meaning of
the plus operator in BGT is to add numbers and strings. Now we will extend, or overload, the plus operator so that it carries the
additional meaning of adding vectors.

The way to overload an operator in BGT is to define a method with a special name. For example, to overload the plus operator
we would define a method called opAdd, and to overload the star operator, we would define a method called opMul. While
we're at it, we can even overload the double equals (==) operator (==) by defining a method called opEquals. Incidentally, we
have already defined all three of those methods, so we simply have to rename them. Here is the revised code with operator
overloading:

class test_vector
{

double x;
double y;
double z;
test_vector(double x, double y, double z)
{
this.x = x;
this.y = y;
this.z = z;
}
double get_x()
{
return x;
}
double get_y()
{
return y;
}
double get_z()
{
return z;
}
string to_string()
{
return "(" + x + ", " + y + ", " + z + ")";
}
test_vector opAdd(test_vector@ other)
{
return test_vector(x+other.x, y+other.y, z+other.z);
}
test_vector opMul(double scalar)
{
return test_vector(x*scalar, y*scalar, z*scalar);
}
test_vector opMul_r(double scalar)
{
return this*scalar;
}
bool opEquals(test_vector@ other)
{
return x==other.x && y==other.y && z==other.z;
}
}

void main()
{
test_vector v1(3, 4, 5);
test_vector v2(5, 4, 3);
test_vector@ v3 = v1 + v2;
alert("For your information", v1.to_string() + " + " + v2.to_string() + " = " + v3.to_string());
test_vector@ v4 = 6 * v1;
alert("For your information", "6 * " + v1.to_string() + " = " + v4.to_string());
if(v1 == v2)
{
alert("For your information", v1.to_string() + " and " + v2.to_string() + " are equal.");
}
else

{
alert("For your information", v1.to_string() + " and " + v2.to_string() + " are not equal.");
}
}

A careful reader may have noticed at this point that I cheated by also defining a method called opMul_r. Many of the magical
method names used for operator overloading, such as opAdd or opMul, also have counterparts with _r appended. The r stands
for reverse, and all it does is tell BGT that we wish to overload the same operator but with operands swapped. In this case, the
method opMul defines what happens when an expression like v*5 is evaluated for a vector v, and the method opMul_r defines
what happens when an expression like 5*v is evaluated.

We can categorize operators by their number of operands. Operand is the technical term for a value upon which an operator
acts. For example, in the expression x+y we have + as the operator and x and y as operands.

An operator with only one operand is called a unary operator. To overload a unary operator, you define a method with no
parameters because the only operand is the object itself. Here are the two magic method names for the unary operators which
can be overloaded:

opNeg overloads -
opCom overloads ~
opPreInc overloads ++ (prefixed)
opPreDec overloads -- (prefixed)
opPostInc overloads ++ (postfixed)
opPostDec overloads -- (postfixed)

Next we come to the binary operators and a somewhat longer list. A binary operator, as you may have guessed, is one which
acts upon two operands. To overload a binary operator we define a method which takes one parameter because the other
operand is the object itself. Here is the list of magic method names in all its glory:

opAdd and opAdd_r overload +
opSub and opSub_r overload -
opMul and opMul_r overload *
opDiv and opDiv_r overload /
opMod and opMod_r overload %
opAnd and opAnd_r overload &
opOr and opOr_r overload |
opXor and opXor_r overload ^
opShl and opShl_r overload <<
opShr and opShr_r overload >>
opUshr and op_Ushr_r overload >>>

The following operators are also binary but they are considered different as they are the so-called assignment operators of BGT.
An assignment operator is one which changes the value of a variable or property. For example, when we write the simple
statement
i = 5;
we have used the assignment operator =. Here is the corresponding list:

opAssign overloads =
opAddAssign overloads +=
opSubAssign overloads -=

opMulAssign overloads *=
opDivAssign overloads /=
opModAssign overloads %=
opAndAssign overloads &=
opOrAssign overloads |=
opXorAssign overloads ^=
opShlAssign overloads <<=
opShrAssign overloads >>=
opUshrAssign overloads >>>=

Next we will cover what is known as the index operator. You have probably already seen how it is used only at that time you
may not have been aware of the fact that it is an operator. It is the syntax you use for retrieving an element from an array by
following the array name with a pair of brackets within which you specify an index. For example, if we had defined an int array
called a, the expression a[3] would be an instance of using the index operator.

To overload the index operator, we define a method called opIndex. It takes the index as parameter and returns the value at the
given index. Here is an overloaded index operator for the vector class:

double& opIndex(int i)
{
if(i==0)
{
return x;
}
else if(i==1)
{
return y;
}
else if(i==2)
{
return z;
}
}

An important detail to notice about the above code is the return type. Rather than merely returning double, this method is
returning something called double&. Recall that we discussed the & symbol before when we introduced the concept of
references. This method returns a reference to a double.

Returning a value by reference is a way to tell BGT not to make a copy of the value but instead return its location, or address, as
if a handle had been used. An additional advantage of a reference, however, is that it can be used on the left side of an
assignment. This means we can now change the y component of a vector v by writing
v[1] = 42;

Six operators are left. All of them are binary, and all of them calculate a yes/no answer to a specific question about the relation
between two values. This is why we call them relational operators. Another common term for them is comparison operators
because they are commonly used to compare values.

We begin with the equality operators == and != which answer the question whether two values are equal or not equal,
respectively. To overload those two operators we need only define the single method opEquals. It must take exactly one
parameter, and it must also return a bool value.

The four remaining operators are <, <=, >, and >= which answer the question whether the first operand is less than, less than or
equal to, greater than, and greater than or equal to the second operand, respectively. All four of them can be overloaded in one
fell swoop by defining a method called opCmp. This method should take exactly one parameter, and it should return an int value

according to the following rules:

Return zero if the two operands are equal.
Return a negative number if the first operand is less than the second.
Return a positive number if the first one is greater than the second.

12. Handles to functions

As you have learned in the two previous chapters, handles provide a way of passing values back and forth without making
copies of them. Instead of the actual value we merely pass a handle which is, in effect, much like a street address. It is not the
value itself but it contains all the information required for finding the value.

It is indeed possible to store and pass a handle to a function. This is a two-step process. The first step is to define a type for the
kind of function you are going to refer to via handles. By kind of function we mean the same concept which we previously
defined as signature. Any two functions taking the same number and types of parameters and returning the same type of value are
said to share the same signature, and thus can be referred to as the same kind of function.

Here is how to define a function type:

funcdef void my_function_type();

The second step is to use the function type just as you would use any other type when working with handles. Just as with any
other type, the at sign (@) is used to denote a handle. The following example illustrates how to pass a function via its handle to
another function:

funcdef void my_function_type();

void do_n_times(my_function_type@ what, uint n)
{
for(uint i=0; i<n; i++)
{
what();
}
}

void print_a_message()
{
alert("Message", "The answer is 42.");
}
void main()
{
do_n_times(print_a_message, 3);
}

The interesting part of this source code is the function do_n_times. It takes a handle to a function and a number as parameters,
and its effect is to execute the given function the given number of times.

Handles to functions are especially useful when writing code which you know will execute some external function at some point,
but you have no way of knowing, at the time of writing, which function will be executed. It might not even be the same function
for every run of your program. In most cases, you will even know the function's signature, just not the actual implementation
behind it. Should you be faced with a problem of that kind, remember that your code need not be hard-wired to execute any
particular function but may instead use a handle to any function the user of your code may wish to define.

13. Using multiple scripts

One very useful feature of BGT is the ability to combine multiple scripts into one program. This can be handy if you are writing a
large game, since it is very time consuming to look through one script for a certain function. Therefore, we have the ability to
categorise our functions spanning multiple scripts, such as a script for menu functions, a script for the player, a script for the AI,
another for sound management, etc. In this way, your main script may only be 200 lines long, for example, instead of 1500.
To use multiple scripts, you have to include them in your main script with the following statement:

#include "scriptname.bgt"

When this happens, the engine will read this script as if it were part of the main script, and any variables, functions or classes that
you use can either be present in the main script or in one of your includes. Because of this, it is not necessary to add a main
function in included scripts, since once the compiler gathers the data from all the included scripts, the main function is going to be
in your main script. Therefore if you put the main function in other included scripts you would have multiple void main() functions
and the compiler will flag an error.
When you include a script specifying a relative path, it will first search for the script in the current working directory. That will
usually be the directory in which your script is stored. If the script cannot be found, it will look in the BGT includes directory. If
the script still cannot be found, the engine will raise an error.
The end user does not need these include files, since all your included scripts will be packaged in with your program once it is
compiled.

Final notes

Well, that's more or less everything. If you have gotten this far, then you should have everything you need to get started. At this
point, I would strongly recommend thinking about everything you have learned and going back and re-reading any sections or
parts that don't make sense to you. Play around with the examples provided in the tutorial so you become familiar with them, and
then feel free to make your games known to the world. The BGT help file comes complete with a full set of reference guides for
functions and objects, with fully functional examples, which you can refer to at any point if you are stuck.
All I can do now is wish you the best of luck, and happy coding. Hope to see your first hit available soon!

Memory Train

In this first tutorial we will develop a game called "Memory Train." No, it will not involve motorized strolls down Memory Lane,
and neither will we deal with wagonloads of microchips. This game will train your memory, hence the title.

Contents

1. An overview of the game creation process
2. Designing Memory Train
3. Implementing Memory Train

3.1 Preparations
3.2 The smallest possible BGT script
3.3 A script that does something
3.4 Making decisions
3.5 A menu for Memory Train
3.6 Sound and speech
3.7 Time
3.8 Keyboard input
3.9 The main function
3.10 Keyboard practice
3.11 Playing a tone
3.12 Playing the game
3.13 Playing a sequence
3.14 Testing the player's memory
3.15 Putting it all together

1. An overview of the game creation process

The story of every game begins with an idea, an initial metaphorical spark in the designer's brain. This is the one creative impulse
out of which the entire project evolves. It is vital that we hold on to that initial idea throughout the entire process of designing,
implementing, and testing. Only when we hold on to that first creative impulse will we later be able to determine if we are still on
track and take corrective measures if it turns out that we are not. Such an initial idea will usually fit into one small paragraph of
text, often a single sentence.

Before we dive into programming tasks, let us talk briefly about the process that leads from that initial idea to the finished game.
This process involves a number of activities, including gameplay design, story design, sound design, implementation, and testing.
For the purposes of describing the game creation process, let us assume that each of these activities is carried out by another
person.

The central role is played by the gameplay designer because her job is to create the high-level design of how the game will
operate. If we were to design a card game, for instance, the gameplay designer would primarily concern herself with creating the
rules of the game. Here are some questions the gameplay designer would need to ask herself: How many cards does the game
use? Will it be a trump game? Will there be suits and ranks? How will the score be determined? How long should the typical
game take? Will it be possible to play against the computer? Against a human opponent at the same computer? Over the
network? All of these questions can be answered without drawing a single picture, without recording a single sound effect, and
without writing a single line of code.

In determining the high-level design, the gameplay designer generates the input required by most of the other team members. For
example, the programmer needs to know the exact rules of the game in order to be able to translate them into code.

Eventually, the combined effort of programmer and sound designer will culminate in a finished prototype. This is in turn closely
inspected by the gameplay designer to ensure that the finished product will meet her expectations. In case the prototype does not
live up to the gameplay designer's expectations, the appropriate changes will be performed by the various team members and a
new prototype is created. This loop continues until the gameplay designer determines that the resulting prototype is in fact the
game she designed. Finally, as with any piece of software, the game will have to go through the various stages of alpha and beta
testing to ensure its quality. Alpha testing is performed behind closed doors by a dedicated team of testers, while beta testing is
performed by potential customers volunteering to give your product a try before it is finally released.

2. Designing Memory Train

For this project the initial idea is as follows:

The objective in our game is to repeat sequences of tones which get more complex as the game progresses.

Here's how it works: The game is played using the arrow keys, and each of the four arrow keys is associated with a specific
tone. When the game begins, the computer randomly plays one of the four tones, and the player has to hit the corresponding
arrow key. Next, the computer plays that first tone again and then adds a second tone, and the player has to hit the two
corresponding arrow keys in sequence. If the player gets it right, the computer adds yet a third tone, and the player has to repeat
all three tones in sequence by hitting the corresponding keys. This process continues until the player makes a mistake by either
pressing the wrong key or failing to react within a certain amount of time. When the player makes a mistake the computer will
play a special sound after which the final score will be announced. The final score is the number of tones which the player was
able to repeat in sequence.

Here's an example:
The computer plays the tone for the up arrow key.
The player hits the up arrow key.
The computer plays the tones for up arrow and left arrow, one after the other.
The player hits up arrow then left arrow.
The computer plays the tones for up arrow, left arrow, left arrow.
The player hits up arrow, left arrow, left arrow.
The computer plays the tones for up, left, left, down.
The player presses up arrow, right arrow, thereby making a mistake.
The game ends, and the final score is 3 points because the longest sequence the player successfully repeated consisted of three
tones.

A key aspect of successfully playing this game is memorizing which arrow key is tied to which tone. To make this as easy as
possible we will implement a keyboard practice mode in which the player can press the arrow keys and listen to the tones they
generate without any time constraints. To enter this practice mode, the player selects the appropriate option from a menu. Not
only will this make our game look considerably more professional but it will also allow me to explain how to create menus in
BGT. And if you thought it would end there, let me inform you that our menu will also have background music.

Exercise:

Reread the above game design and create a list of the sounds the game will contain.

Here is my solution to the above exercise:
- Four sounds for the four different tones
- An error sound to play when the player makes a mistake
- A music loop to play when the menu is active

Since this is a tutorial about programming rather than sound design, you will need to create the sounds that you wish to use for
the game. In the below code, we have selected the filenames that we use for each of these sounds. The four tones are named
1.wav, 2.wav, 3.wav and 4.wav, the error sound is named error.wav, and the music loop is named, appropriately enough,
music.wav.

3. Implementing Memory Train

3.1  Preparations

It is almost time to actually start programming! But before diving into source code, let us get some preliminaries out of the way.

Given the fact that you are reading this tutorial, it is quite likely that you have already installed BGT on your computer. But just in
case you haven't, now is the time to do so.

Next, please create a new, empty directory for this project, and place the sounds you created inside this directory.

Finally, open your favorite text editor and create an empty text file called memory_train.bgt in the newly created directory. Some
text editors, including Notepad, have a tendency to add a .txt extension to any file name that does not already end in .txt. A tried
and tested remedy for this is to include the file name in double quotes. Another way to prevent the automatic .txt extension is to
choose "all files" in the file type field before clicking the "Save" button.

3.2 The smallest possible BGT script

If you have studied the language tutorial, you will know that every BGT script contains one or more functions. The word
"function" comes from the Latin "functio" which means "execution" or "performance." In programming, a function is some code
that performs a well-defined task.

A function, in performing its task, can call other functions to carry out subtasks. The important thing to keep in mind here is that
when one function calls another, the calling function is not finished. This leads to the rather peculiar fact that several functions may
be executing at the same time but only one of them is in control. As an example from daily life, let us assume that the function
"get_dressed" has called the function "put_on_socks". Now the function "put_on_socks" is in control but the function
"get_dressed" is not finished. Instead, it is patiently waiting in the wings for "put_on_socks" to finish. As soon as this happens,
"get_dressed" is back in control and continues executing where it left off, which will be just after the call to "put_on_socks".

I strongly urge you to reread the previous paragraph until you understand it completely. One of the main reasons why people find
programming confusing is that they have misunderstood the concept of functions and function calls. In case you are still confused,
the next time you put on your socks you will feel a warm glow inside and the concept of function calls will make perfect sense.

As an added bonus, once a function finishes it may return a value to its caller. This is appropriately called the function's return
value, or simply its result.

It is well worth our time to take a look at the smallest BGT script which could possibly be written. For although it stops almost
immediately after it starts and does exactly nothing in between, it yet provides the structure, or framework, of any other BGT
script including Memory Train. Since it will eventually become part of Memory Train anyway, now is also the time to copy it into
your memory_train.bgt file. Here it is, in full:

void main()
{

}

This defines a single function called main. The word "void" specifies that this function has no return value, which makes sense
once you realize that it is the solitary function in this script and thus has nobody to return anything to. But its lack of a return value
has a much more practical reason, and this has to do with the special significance of the name "main". The main function is special
in that it is where execution of every BGT script begins, and the execution of every BGT script ends when the main function
finishes. The two braces you see above embrace between themselves the entire execution of your script.

3.3 A script that does something

Let us now make our script actually do something by placing some instructions between those braces.

Exercise:

Find the alert function in the BGT reference, and with the help of the reference modify your script to display a message box. The
title of the message box should be "Important Information", and the message text should be "Hello, I am John Doe, and I am a
programmer." Instead of John Doe, put your own name into the message.

Hint on using the reference: The "alert" function is part of BGT's foundation layer.

Your script should now look something like this:

void main()
{
alert("Important Information", "Hello, I am Jane Smith, and I am a programmer.");
}

To execute your script, simply save the file, locate it in Windows Explorer and hit enter on it. Note that this only works when
BGT is installed.

In case you didn't notice, the line:
alert("Important Information", "Hello, I am Jane Smith, and I am a programmer.");
is a function call. The function being called is "alert", and the calling function is "main". You can see that to call a function, we
write its name, followed by a pair of parentheses in which we specify the information we would like to pass to the function, and
finally a semicolon. In this case we pass two pieces of information to the "alert" function: a title and a message. There is a
technical term for pieces of information passed to a function, by the way. They are called parameters.

Let's see what happens if we add another function call. Our new script is as follows:

void main()
{
alert("Important Information", "Hello, I am Jane Smith, and I am a programmer.");
alert("How many roads must a man walk down?", "The answer is 42.");
}

When you run this script, you will notice that, as you might expect, it displays two message boxes, one after the other. This
serves to illustrate the point I made above, namely, that the main function is not stopped when it calls the alert function. As soon
as the alert function has finished doing its first job, control flows back into the main function which then proceeds to call the alert
function again.

3.4  Making  decisions

A BGT script can do much more than carry out instructions in sequence. Sooner or later our scripts must learn to make decisions
based on what the user does. Only then can we develop interactive scripts, and interactivity is, after all, what differentiates a
game from a play.

Look at the following script:

void main()
{
question("A personal question", "Do you believe in the flying spaghetti monster?");
}

When you execute this script, it displays a message box similar to the one displayed by a call to alert. However, this time the
solitary OK button is replaced by a Yes and a No button.

Exercise:

Read the entry on the question function in the BGT reference.

Asking the user a question without reacting to the answer is pretty pointless, so we will now modify the script to make a decision
based on which of the two buttons the user clicks. As you learned by consulting the reference, the "question" function returns a
value of 1 for the Yes button, 2 for the No button, and 0 if an error occurs.

To keep track of this return value, we will introduce a variable. A variable is a named container for a value. It is called variable
because the value it contains may change.

To define a variable, we write its type followed by its name followed by a semicolon, like so:
int answer;

This defines a variable named "answer" which can hold a value of type "int". "int" is shorthand for "integer" and means a whole
number, i.e. a number without a decimal point. If you would like to learn about the various other types supported by BGT, let me
refer you to the language tutorial.

When we have defined a variable, we can assign a value to it by writing the name of the variable, an equals sign, the value we
would like to assign, and finally a semicolon:
answer = 42;

In our running example, we don't want to place the number 42 in our variable. Instead, we would like our variable to store the
value that the "question" function returned. We achieve this by simply replacing our 42 with the call to the "question" function:

void main()
{
int answer;
answer = question("A personal question", "Do you believe in the flying spaghetti monster?");
alert("Thank you!", "Your answer was " + answer + ". See you later.");
}

If you try out this script, you will notice that you get different messages depending on which of the two buttons you choose. This
is because the variable "answer" receives the value returned by the "question" function. So this is in fact our first interactive script.

Note also how we used the plus sign to tie several things together into something larger:
"Your answer was " + answer + ". See you later."
Scanning this from left to right, we find some text in double quotes, a plus sign, a variable name, another plus sign, and some

more text in double quotes. In this case, the plus signs serve to tie these three things together into the message we would like to
display, like tying together three strings of pearls into a longer string of pearls. Programmers refer to a piece of text as a string
because it is easily visualized as a string of characters. To use this term in context, we have just added a string, a number, and
another string to form our message, which is itself a string.

You might be wondering, if the plus sign is used for tying strings together, how would you add two numbers? The surprising
answer is that the plus sign is used for that purpose as well. The plus sign is a symbol which may refer to different activities,
depending on context. In technical terms, the plus sign is an overloaded operator.

In our running example, our script is now able to react differently depending on which button the user clicks, but our reactions
are not yet very meaningful. Wouldn't it be great if we could display completely different messages depending on the user's
response? The following script does exactly that:

void main()
{
int answer;
answer = question("A personal question", "Do you believe in the flying spaghetti monster?");
if(answer == 1)
{
alert("How interesting!", "Thank you for being that honest.");
}
else if(answer == 2)
{
alert("I thought so!", "Now don't tell me the invisible pink unicorn got you first.");
}
else
{
alert("Whoops!", "Something is dreadfully wrong here! Maybe it's the monster taking revenge.");
}
}

This script uses the "if" statement to make a decision based on the value of the "answer" variable. If the value is 1, the first
message is displayed. Otherwise, if the value is 2, the second message is displayed. Finally, if the value is neither 1 nor 2, the
third message is displayed, indicating that an error has occurred.

Keep in mind that the keyword "if" is always followed by a condition in parentheses, and that in this case the closing parenthesis
is never, I repeat, never, followed by a semicolon.

A condition is something which can either be true or false. This is in line with the usage of the word in every-day English. For
example, the condition for going by train is that you are in possession of a valid ticket. If the condition is false you may not board
the train.

The condition to test whether two things are equal looks like this:
a == b
Note the use of a double equal sign. This is required because the single equal sign is already used for something else.

Exercise:

What is the single equal sign used for? Hint: The answer is contained earlier in this tutorial.

Note that we have enclosed every branch of the above "if" statement in its own pair of braces. Strictly speaking, this is necessary
only if a branch consists of multiple instructions. Since all three branches in the above script consist of a single instruction, namely,
a call to "alert", we could just as well have written the following:

void main()
{
int answer;
answer = question("A personal question", "Do you believe in the flying spaghetti monster?");
if(answer == 1)
alert("Ramen to you, then!", "Thank you for being that honest.");
else if(answer == 2)
alert("I thought so!", "Now don't tell me the invisible pink unicorn got you first.");
else
alert("Whoops!", "Something is dreadfully wrong here! Maybe it's the monster taking revenge.");
}

Over the years of my programming career I have formed the habit of always including the braces from the start even in such
cases where they are not strictly necessary. This way, when I add more statements at a later date I never have to worry about
whether or not to add another pair of braces.

Just so that you know, the braces in a function definition are mandatory. For instance, the definition of your main function will
always contain braces even if the function consists of a single instruction---in fact, even if it should consist of no instructions at all!

Exercise:

Read the relevant parts of the language tutorial to familiarize yourself with the various programming constructs for making
decisions. In particular, pay close attention to the if, while, and for statements as they will be used in this tutorial.

3.5 A menu for Memory Train

In our quest for theoretical background we have digressed quite a bit from our original endeavor, which was to implement the
game we designed in the previous chapter. Now is the time to get back to it.

Let's go back to our simplest of scripts, which was this:

void main()
{
}

Exercise:

The helper layer contains a helpful tool for creating menus. Can you find it in the reference?

The tool we are after is called dynamic_menu. It allows us to easily put together a menu in which the player can select an item
with the up and down arrow keys and activate it by pressing enter. As you undoubtedly know, such menus abound in
audiogames, with typical menu items being "start game", "test speakers", "options", and "exit game".

Let's take a moment to summarize what you already know about functions and variables.

A BGT script contains one or more functions.
A function is a piece of code that performs a well-defined task. Note that code is just another word for program text.
Every script contains a special function called "main". Execution of your script begins at the start of this function and ends
when this function has finished executing.
Functions can delegate subtasks to other functions.

When function f asks function g to perform its task, we say that function f calls function g.
When f calls g, f may pass some pieces of data to g. These pieces of data are called parameters.
When g has finished, it may pass a single piece of data back to f. This is called a return value.

A variable is a named container for a value which may change.
Every variable has a type, and while the value of a variable may change, the type never will. This means that an int variable will
only ever store values of type int, and a string variable will only ever store strings. This may sound awfully restrictive, but it also
means less work for the computer and fewer opportunities for programming errors. Besides, why should you want to use a single
variable for vastly different purposes when you can have as many variables as you like?

In this section we will go one step further and talk about a kind of variable which has other variables and functions living inside it.
A value with variables and functions inside it is called an object, and a variable holding an object is called an object variable, just
as a variable holding an int is called an int variable. If this all sounds just a bit confusing, let me assure you that it will quickly
become second nature to you once you see it in action. In just a few paragraphs you will realize that objects are all about
simplicity.

Examine the following script:

#include "dynamic_menu.bgt"

void main()
{
show_game_window("Memory Train");
dynamic_menu menu;
menu.add_item_tts("Start game");
menu.add_item_tts("Keyboard practice");
menu.add_item_tts("Exit game");
menu.allow_escape = true;
menu.wrap = true;
menu.run("Please choose a menu item with the arrow keys, then hit enter to activate it.", true);
}

Exercise:

With the help of the reference, try to figure out what the above script will do. Test your hypothesis by running the script. Note:
This script contains some constructs we have not covered yet, but they will all be explained below.

Let's start with the elements you already recognize. First, you will immediately have noticed that this is your standard main
function with a sequence of instructions between the braces. Next, examine the following line:
show_game_window("Memory Train");
This is some name followed by a pair of parentheses with some data between them, and finally a semicolon. What do we call
such a thing? Yes, a function call.

Exercise:

How many parameters does the above function call have?

Exercise:

Find out what the function does.

You may be wondering why we would want to display a window in the first place when our game will not have any visual

elements. The answer is that on Microsoft Windows, any program that handles keyboard input should display a window. Our
game will only be able to react to keyboard input when our game window is active. This will also allow the player to switch back
and forth between our game and other running applications.

Next, take a close look at the following line:
dynamic_menu menu;

If this confuses you, it might be helpful to realize that this is structurally similar to something we have seen earlier:
int answer;

Both of these statements are variable definitions. One of them defines a variable of type int and gives it the name "answer", the
other one defines a variable of type dynamic_menu and gives it the name "menu". And just as int is a data type for numbers,
dynamic_menu is a data type for menus. The word "dynamic" is used here meaning "flexible" because you, as game programmer,
can decide which items the menu will contain.

Let's go ahead and add some items to our menu:
menu.add_item_tts("Start game");
menu.add_item_tts("Keyboard practice");
menu.add_item_tts("Exit game");

Notice the striking similarity to function calls? The only difference is that this time we are calling a function that lives inside our
menu variable. Recall that an object is a value with variables and functions living inside it. In this case we have on our hands an
object of type dynamic_menu.

To call a function inside an object, we write the variable name, a full stop, and then the name of the function we are calling. This
is followed by the usual pair of parentheses which, as we have learned, is part of every function call. A function inside an object
is called a method. To use this term in context: add_item_tts is a method of the menu object.

An object gets its methods from its type. Thus, to find out what methods an object provides we need only know what type, or
class, the object belongs to. The "menu" object is of class dynamic_menu, so we need only consult the reference of
dynamic_menu to find out about the available methods.

Exercise:

The following statement defines an object variable:
file saved_game;
Which of the following statements are valid?
saved_game.open("test.dat", "w");
saved_game.print("Hello, world!");
saved_game.write("Hello again!");
saved_game.close();
saved_game.open("test2.dat");

With the menu set up the way it is, we need to perform one more step to present it to the user:

menu.run("Please choose a menu item with the arrow keys, then hit enter to activate it.", true);

3.6 Sound and speech

This section covers two additional building blocks required for putting Memory Train together. Let us begin with sound, which
will be an essential feature of any audio game you will ever develop.

In BGT, every sound is represented by a sound object. One way of visualizing a sound object is to picture it as a tape player
with its various controls. For example, while your tape player probably has buttons labelled play, stop, and pause, as well as a
volume knob or slider, the sound object has methods named play, stop, and pause, as well as a volume property. In addition, a
program sometimes needs to check if a sound is still playing. To allow this, the sound object has a property called playing which
will be true as long as the sound is playing, and will become false as soon as it has stopped. The following example illustrates
how the sound object can be used:

void main()
{
sound intro; // Creates the sound object.
intro.load("intro.wav"); // Loads the sound; equivalent to putting a tape into the player.
intro.play(); // Starts playing.
while(intro.playing)
{
// Do something while the intro is playing.
wait(5); // Give other Windows tasks 5 milliseconds time.
}
// The sound has stopped.
}

Some games, including Memory Train, make use of a speech synthesizer on the player's computer. The way to do this in BGT is
to use a tts_voice object, where tts stands for "text to speech." The two most interesting methods of tts_voice are speak and
speak_wait. The speak method will begin speaking the string which was passed to it and will continue speaking in the
background while your program may do other things. In comparison, the speak_wait method will additionally wait until the
speech has stopped, and only then will your program continue with the next instruction.

3.7 Time

All but the simplest of games need to keep track of time in some way. BGT provides a timer object for this purpose. The most
important features of the timer object are the restart method and the elapsed property. By calling the restart method, your
program is able to reset the timer back to zero in much the same way as if you were restarting a stopwatch. The elapsed
property will always contain the number of milliseconds since the timer was last restarted, or, if it was never restarted, elapsed
will contain the number of milliseconds since the timer was created.

It is considered good behavior for a Windows application to regularly wait for a few milliseconds. Not only does this give other
processes in the system more time to execute their own tasks but it also increases the system's ability to save energy. In BGT,
you can use the wait function to put your program to sleep for the specified number of milliseconds. Note that the time which
actually passes may differ from the time you specified, so it is advised that you use a timer to check how many milliseconds have
actually passed. When in doubt, do not use the wait function to time game events, but use a timer instead as timers are usually
accurate to the millisecond.

3.8 Keyboard input

In order for your game to be able to respond to keyboard input, BGT contains two functions which let you check the status of
any given key on the user's keyboard. Use the key_down function to find out if a given key is being held down at that moment.
Note that key_down will return true as long as the key is being held down. If you are not interested in how long a key is being
held but rather would like to be informed of every keypress just once, use the key_pressed function instead. When a key is held
down, key_pressed will return true only the first time you check that particular key, and false on subsequent calls. Only if the key
was released and is now being held down once more will key_pressed return true again.

To find out which keys can be checked and how they are expressed in BGT, consult appendix A of the BGT help system.

3.9 The main function

Armed with a basic understanding of the fundamental features of BGT, we can now begin coding the bulk of Memory Train. Let
us start with the main function which, as you learned earlier in this tutorial, is the heart of any BGT script. The responsibilities of
the main function are as follows:

1. It loads the various sounds required by the game into memory.
2. It sets up the game menu.
3. It speaks a short intro.
4. It repeatedly runs the menu and executes the function the user has chosen.
5. The repetition ends as soon as the user presses the escape key in the menu or chooses the "exit game" entry.

While we are at it, we will also define some global variables the main function refers to. Our code now looks as follows:

#include "dynamic_menu.bgt"

// Sound objects.
sound music;
sound error_sound;
sound[] tone(4); // The four tones used for sequences.

void main()
{
// Load the four tones.
tone[0].load("1.wav");
tone[0].volume = -10;
tone[1].load("2.wav");
tone[1].volume = -10;
tone[2].load("3.wav");
tone[2].volume = -10;
tone[3].load("4.wav");
tone[3].volume = -10;

// Load the error sound.
error_sound.load("error.wav");
error_sound.volume = -10;

// Load the music.
music.load("music.wav");
music.volume = -10;

// Set up voice.
tts_voice voice;

// Set up menu.
dynamic_menu menu;
menu.allow_escape = true;
menu.wrap = true;
menu.add_item_tts("Start game");
menu.add_item_tts("Keyboard practice");
menu.add_item_tts("Exit game");

// Show game window and speak welcome message.

show_game_window("Memory Train");
voice.speak_wait("Welcome to Memory Train!");

// Loop the music in the background before running the menu.
music.play_looped();

int choice; // This stores the user's menu selections.

do
{
choice = menu.run("Please choose a menu item with the arrow keys, then hit enter to activate it.", true);
if(choice==1)
{
music.stop();
play_round(); // We will define this function later.
music.play_looped();
}
else if(choice==2)
{
music.stop();
keyboard_practice(); // We will define this function later.
music.play_looped();
}
}
while(choice!=0 and choice!=3);

// The user pressed escape or chose to exit.
voice.speak_wait("Thanks for playing.");
}

3.10 Keyboard practice

If the user chooses the keyboard practice item from the menu, our above main function will call a function named
keyboard_practice. This has the following responsibilities:

1. It speaks instructions on how to operate the keyboard practice mode.
2. It waits for the user's keypresses.
3. When an arrow key has been pressed, it plays the appropriate tone.
4. When escape has been pressed, keyboard practice is aborted.

Here is the code for the keyboard_practice function:

void keyboard_practice()
{
tts_voice voice;
voice.speak_wait("Press the arrow keys to find out which key generates which tone.");
voice.speak_wait("Press escape to stop practicing.");
while(!key_pressed(KEY_ESCAPE))
{
if(key_pressed(KEY_LEFT))
{
play_tone(0);
}

else if(key_pressed(KEY_DOWN))
{
play_tone(1);
}
else if(key_pressed(KEY_RIGHT))
{
play_tone(2);
}
else if(key_pressed(KEY_UP))
{
play_tone(3);
}
wait(5);
}
}

3.11 Playing a tone

Notice how we introduced the play_tone function in the above code for keyboard_practice. The responsibility of play_tone is to
play a single tone. Here is the corresponding code:

void play_tone(int i)
{
tone[i].stop();
tone[i].play();
}

This function first stops the tone in case it is already playing, then restarts playing it. By the way, if the expression tone[i] confuses
you, this might be a good time to read up on the subject of arrays in the language tutorial.

3.12 Playing the game

If the user chooses the "start game" item from our game menu, the main function calls the play_round function. This function is
responsible for carrying out an entire round of play, after which the score is announced. Here is the code for play_round:

void play_round()
{
// Initialize game state.
bool game_over = false; // The game will continue as long as this is false.
int[] sequence; // The running sequence.
int sequence_length = 0;
float time_between_tones = 500; // The initial speed at which tones are played, in milliseconds.
float time_between_inputs = 2000; // Maximum time to input next key before player gets bounced.
do
{
// Add another tone to the sequence.
sequence_length++;
sequence.resize(sequence_length);
sequence[sequence_length-1] = random(0,3);
// Play back the sequence from start to finish.
output_sequence(@sequence, time_between_tones);
// Let them repeat it if they can.

game_over = input_sequence(@sequence, time_between_inputs);
// Every time another five tones have been mastered:
// increase the speed ever so slightly.
if((sequence_length%5) == 0)
{
time_between_tones = time_between_tones*0.9;
// Make sure it does not fall below 150.
if(time_between_tones < 150)
{
time_between_tones = 150;
}
}
}
while(!game_over);
int score = sequence_length-1; // minus 1 because they failed on the last.
tts_voice voice;
voice.speak_wait("Your final score was " + score);
}

3.13 Playing a sequence

The output_sequence function plays a sequence of tones so that the player may try to memorize it. It requires two parameters,
the sequence to output, and the time between the tones. This flexible approach was taken to enable the game to increase the
speed over time. Here is the code for output_sequence:

void output_sequence(int[]@ sequence, float time_between_tones)
{
timer clock;
int current;
for(uint i=0; i<sequence.length(); i++)
{
// For every tone but the first, wait before playing.
if(i>0)
{
clock.restart();
while(clock.elapsed < time_between_tones)
{
wait(5);
}
}
current = sequence[i];
play_tone(current);
}
// Finally, wait for the last sound to subside.
while(tone[current].playing)
{
wait(5);
}
}

3.14 Testing the player's memory

The final puzzle piece is the input_sequence function. This function is responsible for testing if the player remembers the sequence
correctly. If the player fails to press one of the arrow keys in a given time, or presses the wrong arrow key, this function will
return true to indicate that the game is over. If, on the other hand, the player manages to replay the sequence correctly, then this
function returns false, indicating that the game is to continue. Here is the code for input_sequence:

bool input_sequence(int[] @sequence, float time_between_inputs)
{
timer clock;
for(uint i=0; i<sequence.length(); i++)
{
// Do the following for every tone in the sequence:
clock.restart();
int input = -1; // Set it to something invalid.
while(clock.elapsed < time_between_inputs)
{
if(key_pressed(KEY_LEFT))
{
input = 0;
}
else if(key_pressed(KEY_DOWN))
{
input = 1;
}
else if(key_pressed(KEY_RIGHT))
{
input = 2;
}
else if(key_pressed(KEY_UP))
{
input = 3;
}
if(input>=0)
{
break; // Stop waiting because something was typed.
}
wait(5);
}
// Game is over if timed out or wrong key.
if(input!=sequence[i])
{
error_sound.play_wait();
return true;
}
// Play back successful tones as feedback
play_tone(input);
}

// The player managed to replay the entire sequence.
// Wait one second to let them catch their breath.
clock.restart();
while(clock.elapsed < 1000)
{
wait(5);
}
return false;
}

3.15 Putting it all together

The complete source code for Memory Train is contained in the code fragments given in sections 3.9 to 3.14. In order to test the
game, you can simply paste all of the code fragments into the file memory_train.bgt which you created earlier. To run the game,
simply navigate to the file memory_train.bgt in Windows Explorer, then press enter to start.

Final exercises:

1. Extend the game to provide a two-player mode. Both players sit at the same keyboard. Player one uses the arrow keys
to play, and player two uses the keys a, s, d, and w. In this variant, sequences are not randomly generated by the
computer. Instead, player 1 starts with one tone, player 2 repeats the one tone and adds another, player 1 repeats the two
tones and adds a third, etc. The game ends when one of the two players fails to press a key in time, presses the wrong
key, or presses a key when it was not his or her turn. The player making the mistake loses, and the other player wins with
a score equal to the longest sequence he typed.
2. Extend the game so that it provides different difficulty levels. Note that this is both a design and an implementation
exercise.

Windows Attack

Contents

1. Introduction
2. The name of the game
3. Sound design
4. Game state
5. The sound pool
6. The game loop
7. Auxiliary functions
8. The virus class
9. The main function
10. Exercises

1. Introduction

Welcome to the second installment of Game Programming in Practice, a series of tutorials in which you learn about core BGT
concepts by seeing how they are used. As you follow these tutorials you create actual games while the syntax, structure, and
techniques of BGT programming become second nature to you.

Let me take this opportunity to remind you that errors are an integral part of programming. They happen to the best of us, and
rather than seeing them as frustrating stumbling blocks, you will do much better considering them as opportunities to test and
even improve your logical skills.

Finally, remember to take frequent breaks. Stepping back from a problem for a while and returning to it refreshed might make all
the difference.

With these formalities stored safely in the back of your mind, let's get going!

2. The name of the game

In this tutorial you will create a game called Windows Attack. If you have ever had trouble with computer viruses in the past, you
will be delighted to know that this game finally lets you take sweet revenge.

Here is the basic idea: You are an antivirus program which must target and destroy approaching computer viruses before they
reach and infiltrate the system!

Let's flesh out this basic idea until it becomes a real game.

Computer viruses will fall towards your system from above, making noise as they approach. Your job is to target a virus with the
left and right arrow keys until its sound is in the center of your stereo field, then hit the space bar to trigger your deadly digital
destruction device of doom. If you make it in time, you will score some points depending on how well you performed. However,
if a virus has made it all the way down, it will do some damage to your system. Now, the system can only stand so much damage
until it crashes with the dreaded blue screen of death, at which point the game will be over.

3. Sound design

Exercise

Based on the game's description in section 2, can you come up with a list of required sounds?

Here is the list I created:

A start sound (start.wav)
The sound a virus makes while falling (virus_fall.wav)
The sound a virus makes when landing (virus_land.wav)
The sound a virus makes when it is destroyed (virus_hit.wav)
The sound of your deadly digital destruction device of doom (gun.wav)
An end sound (game_over.wav)

You may save yourself most of the bother of sound design by simply using some of the standard Windows sounds, or you may
record or download a unique collection of sounds for your game. In any case, I strongly recommend that you use the filenames I
listed above as they will be used later on in this tutorial. It is a good idea to create a new, empty folder for your project at this
point and then copy your sound files into this folder. While you are at it, now is also a good time to create the file
windows_attack.bgt. This will contain the entire source code for Windows Attack and will keep us busy for the rest of this
tutorial.

4. Game state

Every game has a game state, which is all the information required to continue the game. For example, the game state for chess
would primarily consist of the positions of all the pieces on the board and also the information whose turn it is. For Uno, the
game state would consist of each player's hand and the order of cards in the stock and discard piles, along with each player's
current score and the information whose turn it is to discard.

Exercise

From the description of Windows Attack in section 2, identify the game state.

Here is the solution I came up with:
- Player's horizontal position on the board
- Viruses on the board, along with the current height of each virus
- Player's current score
- Player's current number of lives

Let us translate this into code:

// Game state
int player_position; // Current horizontal position of player
virus@[] board; // Locations without viruses will contain null
int lives; // Game over if this falls to zero during play
int score; // Current score

If you have completed the first tutorial in this series or read some of the BGT language tutorial, understanding the above code
won't be much of a problem for you. Let us pause briefly, however, to inspect the most complicated of the lines above:

virus@[] board; // Locations without viruses will contain null

The name of the variable declared here is board, and the data type is virus@[]. This is best understood when reading it from right
to left: The brackets tell us that this is an array, the @ sign indicates it is an array of handles, and the word virus specifies the type
of object to which the handles will refer. The virus class will be defined later in this tutorial.

It is a useful habit to define a function which initializes the game state so that whenever we wish to start a new game, we need
only call this function and can rest assured that each game begins in a well-defined initial state.

Here is that function for Windows Attack:

const int board_size = 21;

const int initial_lives = 3;
void initialize()
{
player_position = (board_size-1) / 2; // Center of board
board.resize(board_size);
for(int i=0; i<board_size; i++)
{
@board[i] = null;
}
lives = initial_lives;
score = 0;
}

You might argue that it is unnecessary to explicitly set all the board locations to null. After all, null is exactly the value with which
handle variables get initialized. However, remember that we might want to play multiple games in one session, and so the board
might still contain leftover viruses from the previous game. The lesson here is that a little extra housekeeping never hurts and leads
to well-defined states at key points in the flow of your program.

Also, note that we have defined constants for the board size and initial number of lives. In general, it makes sense to define
constants for values which will eventually be fixed but which you might change from time to time in your development process. It
is of course perfectly valid to write all constants as literal numbers, but the advantage of a named constant is that you can change
its value at the point of its definition and the change will apply to every point in your source code where the constant is used. As a
general rule, any value which is used more than once in your source code should be given a name.

5. The sound pool

In the first part of this series of tutorials, you learned that sounds are played using sound objects, and that the number of sound
objects you need is the number of sounds your game will play simultaneously. For small projects like Memory Train you will
usually manage the creation, utilization and destruction of those sound objects in your own code. But when matters start to
become more complex, such as when lots of sounds need to be created or repositioned in response to events in the game world,
managing all the sound objects yourself can be a nuisance. For these situations, BGT contains a powerful helper class called
sound_pool. It is called a pool because it can manage a large collection of sound objects and use them as needed. For example,
when you order the sound pool object to play a sound, it will look through its pool of sound objects to find one which is currently
inactive, and only if none can be located will a new sound object be created.

All this may sound dreadfully complicated at first, but the good news is that the sound pool class is incredibly easy to use
because you do not need to concern yourself with most of its technical intricacies. You simply order it to play the right sound at
the right time and place, and the sound pool will take care of the rest.

Another killer feature of the sound pool is that it can automatically position the sounds for you. You simply tell it the coordinates
of your sound sources and your listener, which will usually be the player, and once again, the sound pool will take care of the
rest.

Exercise

Consult the BGT reference about the sound_pool class and familiarize yourself with its methods.

One potentially confusing aspect of the sound pool is its use of so-called sound slots, so let us address them right away to
prevent you from forming the wrong model about them.

When you ask the sound pool to play a sound, for example by calling the play_2d method, it will return a number called a sound
slot. The sound slot is simply a number which the sound pool has assigned to the sound, in much the same way as the
government assigns social security cards to citizens when they get their first jobs. The government uses social security numbers to
quickly find or update a person's data. That's why they ask you for this number whenever you call them. In much the same way,
when you ask the sound pool class to update a sound which is already playing, you will need to provide the sound slot which

was returned to you when you initially started the sound. If some of this doesn't seem to make sense, just read on, and all will be
revealed.

6. The game loop

We already dealt with the initialization of the game state. Let us now turn to the game itself while it is in progress.

Traditionally, this is expressed as a loop which drives the action and which is therefore sometimes called a driver loop. The job
of the driver loop is to run until the game is over and to orchestrate the flow of the various components of the game. In other
words, the driver loop is responsible for the movement of time. Let's see what this would look like:

sound_pool pool;
void game_loop()
{
timer virus_act_timer;
timer virus_spawn_timer;
while(lives > 0) // We loop until the game is over
{
player_act(); // Check if player pressed a key and let him move or shoot
if(virus_act_timer.elapsed >= 200) // Viruses move 5 times a second
{
viruses_act();
virus_act_timer.restart();
}
if(virus_spawn_timer.elapsed >= 3000) // New virus every 3 seconds
{
virus_spawn();
virus_spawn_timer.restart();
}
wait(5); // Be nice to other apps on this machine
}
}

7. Auxiliary functions

Our game loop references a number of functions which we still have to define. First, there is player_act, the function which allows
the player to move and to shoot at viruses.

void player_act()
{
if(key_pressed(KEY_LEFT) and player_position>0)
{
player_position--;
}
else if(key_pressed(KEY_RIGHT) and player_position < (board_size-1))
{
player_position++;
}
if(key_pressed(KEY_SPACE)) // Shoot
{
shoot(); // Let's do this in its own function as it is more complex
}
if(key_pressed(KEY_ESCAPE)) // Exit game
{
lives = 0; // Simulate game over

}
pool.update_listener_2d(player_position, 0);
}

void shoot()
{
if(@board[player_position] is null) // Missed
{
pool.play_stationary("gun.wav", false); // Centered, not looping
}
else // Hit
{
board[player_position].die(); // We tell the virus it has died
score++; // Player gets a point
@board[player_position] = null; // No more virus at this position
}
}

Now, let us define the function viruses_act, which gives each virus currently on the board a chance to move.

void viruses_act()
{
for(int i=0; i<board_size; i++)
{
if(@board[i] !is null)
{
board[i].act();
}
}
}

Finally, let us define the function virus_spawn, which randomly selects a location on the board and, if the location is still vacant,
places a new virus there.

void virus_spawn()
{
int location = random(0, board_size-1);
if(@board[location] is null) // Check if it is vacant
{
virus newbie(location); // Create a new virus
@board[location] = @newbie; // Register it with the game board
}
}

8. The virus class

As if the previous two sections hadn't been code-intensive enough, let's complete the picture by defining the virus class.

class virus
{
int height; // Height above ground. When this falls to zero the virus lands
int falling_sound; // Stores the sound slot for the falling sound loop
int location; // Location of virus on the game board
virus(int location) // Constructor
{

this.location = location;
height = 20;
falling_sound = pool.play_2d("virus_fall.wav", player_position, 0, location, height, true);
}
void act()
{
height--;
if(height<=0) // Virus has landed
{
pool.destroy_sound(falling_sound);
pool.play_2d("virus_land.wav", player_position, 0, location, 0, false);
lives--;
@board[location] = null;
}
else
{
pool.update_sound_2d(falling_sound, location, height);
}
}
void die()
{
pool.destroy_sound(falling_sound);
pool.play_2d("virus_hit.wav", player_position, 0, location, height, false);
}
}

9. The main function

Our mission is almost complete. We have defined the game state and its initialization, and we have specified the game loop along
with all its helper functions. Let us conclude by defining the main function which will set all of it in motion.

void main()
{
show_game_window("Windows Attack");
sound start_sound;
start_sound.load("start.wav");
tts_voice voice;
voice.speak("Welcome to Windows Attack!");
start_sound.play_wait();
initialize();
game_loop();
pool.destroy_all();
sound game_over_sound;
game_over_sound.load("game_over.wav");
game_over_sound.play_wait();
voice.speak_wait("Game over! Your score was " + score);
}

Now, the only thing left to do is to place the line
#include "sound_pool.bgt"
at the beginning of your source code to bring the sound_pool class into the compilation.

10. Exercises

Assemble the code fragments from this tutorial into a working version of Windows Attack.

1.
2.

3.

1.
2.

3.
4.

Modify your game so that instead of simply exiting when the game is over, it asks if the player would like to play again,
and continues accordingly.
Modify your game to offer multiple difficulty levels.
Modify your game so that every minute it spawns a supervirus. A supervirus is different in that it requires three shots to
take down, but if taken down it will provide the player one extra life.

The Amulet of Quang

Contents

1. Introduction
2. The name of the game
3. Let's get serious
4. Designing the interface
5. Music
6. Sound effects
7. Speech
8. Sound design
9. Inventory
10. Items
11. Areas
12. Player position
13. Looking around
14. Moving around
15. Object manipulation
16. Game loop
17. Use your imagination

1. Introduction

Welcome to the third and final installment of Practical BGT, the series of tutorials in which you develop actual playable games
while learning the basic concepts and techniques of BGT programming. If you haven't completed the first two installments in this
series, we strongly recommend that you do so before starting out on part III.

Let us take just a few minutes to go over what you learned in the first two parts.

In part I, Memory Train, you took a straight dive into the syntax and basic functionality of BGT. You learned how the execution
of your game program can be described in terms of functions calling other functions and making decisions based on what they
return. More specifically, you saw how to display a game window and ask for keypresses, how to load and play sounds and
background music, how to keep track of time, and, last but not least, how to exit your game.

In Part II, Windows Attack, you took a more systematic approach to game design. You took the description of a game and
extracted from it a list of required sounds, a description of the game state, and finally, the source code of the game itself,
complete with a self-made script class. In addition, you learned to harness the power of a sound pool to position an arbitrary
number of sounds in real-time.

This final installment will provide you with some powerful programming techniques and design approaches to make your life
easier as you begin to work on larger-scale projects. You will probably not learn any new syntactic constructions here, nor will
the mini game you will write be anywhere as spectacular as those you programmed in the first two parts. Instead, this part is
designed to put the tools into your hands which, combined with imagination and creativity, will allow you to create games of high
software quality and consistent user experience.

If anything in this tutorial should appear confusing or unclear, chances are you will find the missing puzzle pieces in the language
tutorial, the BGT reference, or, indeed, in the first two parts of this series. Sometimes it is also a good idea to just continue
reading with dogged determination, and you might find the answer in the next paragraph. In any case, you should take frequent
breaks, maintain a calm, rational, detective-like approach to complexity, and remember that bugs happen to the best of us, and
are valuable lessons only real life can teach.

With that in mind, let the games begin!

2. The name of the game

In ancient and altogether more interesting days, ages before the golden days of yore, the mysterious days of lore, and the
ridiculously long-ago days of nevermore, the Kingdom of Quang stood proudly. Proudly indeed it stood, impervious to the
ravages of slime, until its two supreme rulers, during one of their regular and excessive drinking binges, got into a fateful and
bloody argument about who stole the cookies from the cookie jar. The question was never answered, but such was the
monstrosity of the ensuing war that the mighty Kingdom of Quang fell and sank beneath the sea. The wizards of Quang,
however, having gazed deeply into their crystal balls, knew that tragedy was afoot, and managed to transfer all the knowledge of
their high culture into a powerful magical artefact---the Amulet of Quang.

You, a secret agent of the even more secret world government, have been transported to the vast underground empire of Quang.
Your mission is to recover the Amulet of Quang and transport it back to the surface of the earth.

3. Let's get serious

Here is a more serious description of the game.

The Amulet of Quang is an audio adventure in which the player can explore his surroundings, walk from place to place, and
interact with the various objects which litter the game world. Every area of the game map may provide its own background
music, and objects may or may not emit sound. The player will be informed by means of a spoken message when the main
character spots a new object. The player can obtain a detailed description of an object by examining it. Some objects can be
picked up. The objects the player is carrying are organized into an inventory list which can be accessed at any time during the
game. It is possible to use objects in the inventory list. The game can be paused at any time.

4. Designing the interface

Exercise

Analyzing the description in the previous section, create a detailed specification of the game's interface. The interface consists of
everything the player needs to know in order to successfully play your game. Mostly, this would be the kind of information you
might put into a user manual. Here are a few questions to put you on the right track:

1.
2.
3.
4.
5.
6.
7.
8.

How does the player walk around the map?
When and in which way is the player informed of available objects?
Will there be footstep sounds?
How does the player examine, pick up, or use an object?
How does the player access the inventory list?
How does the player examine or use an object in the inventory list?
How does the player pause and unpause the game?
Finally, how would the player exit the game?

Here is my solution. In case our answers differ, this does not mean that you have got it wrong, but merely that we have chosen to
create slightly, or maybe even dramatically, different kinds of games. The main reason why I am detailing my solution here is to
show you the level of detail required for a user interface specification. In general, the more complete your specification at design
time, the less trial and error at coding time. It is perfectly reasonable at design time to leave certain details open to discussion or
change, but this should happen as the result of deliberate choice rather than unconscious omission. So let us agree for the
moment to stick to my specification, and if afterwards you feel that your solution is worthwhile trying, I shall leave it as an
exercise to you to go back and redo from that point onward. In fact, I highly recommend doing so. Also, do not be put off if it
seems to take days, or even weeks, to get exactly right. This is to be expected.

The player walks around the game map by using the four arrow keys. Pressing and then immediately releasing an arrow key will
move the player exactly one step in the corresponding direction. Holding an arrow key will move the player one step in the
corresponding direction every half second. In this way, a steady motion is established just by comfortably holding down a key,
but an impatient player can still accelerate the process by tapping the key more quickly.

To give the player feedback that motion has indeed occurred, the game will play a footstep sound for every step the player
takes. If, after taking that step, the available directions for the next step have changed, we will play a slightly modified footstep
sound to alert the player to that change. For example, assume the player is walking down a long east-west corridor, and
suddenly comes upon an exit to the north. This would be the time when our modified footstep sound would be played. So we
define two footstep sounds, one of them soft and unobtrusive, the other slightly more demanding, which might be accomplished
by making it slightly louder or higher in pitch.

Let's move on to what we might call the model of visibility. This is the set of rules which determine what objects the player sees
at any given time. Such a model could get arbitrarily complex, taking into account such details as distance and size of objects,
lighting conditions, fog, or other objects which obstruct the view. For this design, however, we will keep it simple by defining just
two basic visibility rules. An object is visible if
1. it has the same x or y coordinate as the player, i.e. it is straight to the north, east, south, or west,
and
2. there is no wall anywhere between the player and the object.

Whenever the player moves such that a new item is spotted, we will provide a spoken message stating what kind of object it is,
in what direction it lies, and how many steps it would take to get to it. Note that we will only speak the objects as they move into
view, not those which were already visible before. For example, if the player took one step north, we wouldn't need to speak
any visible objects to the north or south because they would have been visible before taking the step. If this sounds confusing, I
recommend you reread the two visibility rules above and then form some examples in your mind to illustrate how they work.
Specifically, it is important to understand that taking a step to the north or south can only change what is visible to the east or
west, and vice versa.

A general description of the area at large is spoken when the player enters it, and repeated as requested via the l command.
Note: l as in look.

To examine, pick up, or use an object, the player must first move to that object's location. A special sound is then played to
indicate that an object has been reached, and the object is announced. To examine it, the player uses the x command. The t
command will attempt to pick up the object. Finally, hitting the space bar will attempt to interact with the object in some way.
For instance, hitting the space bar on a door might try to walk through that door, and hitting the space bar on a gong might strike
the gong.

The tab command will cycle through the objects in the player's inventory. To use an object in the inventory, the player first cycles
to this object by pressing tab until the object is announced, and then presses u, as in use. To examine an object in your inventory,
you would first cycle to it by pressing tab repeatedly, and then press i, as in inventory information.

To pause the game at any time, the player can press the p command. The same command also resumes a paused game.

Finally, the escape key will exit the game. The game may also exit when the player wins or dies, but escape is always available to
exit prematurely.

Pressing page up or page down will allow the player to adjust the volume of the music. This is especially useful in conjunction
with spoken messages, because the music might drown out the voice if too loud.

To make the experience of learning the interface as smooth as it could possibly be, we will implement a help feature which will
make our game self-documenting. Help can be accessed at any time by pressing f1, whereupon the game will provide spoken
information about all the possible keyboard commands at that time.

5. Music

Exercise

One important aspect of game music is its ability to gracefully fade in and out. Write a fade function which can be called as
follows:
fade(sound@ noise, double start_volume, double end_volume, double time, bool interruptable);
When called, this function should fade the given sound object from the start volume to the end volume in the given time in

milliseconds. If interruptable is true, the player can interrupt the process by pressing the space bar. The fade function returns true
if the player interrupted the process using the space bar, and false otherwise.

Here is one possible solution:

bool fade(sound@ noise, double start_volume, double end_volume, double time, bool interruptable)
{
noise.volume = start_volume;
timer t;
double elapsed = t.elapsed;
while(elapsed < time)
{ noise.volume = start_volume + (end_volume-start_volume)*elapsed/time;
wait(10);
elapsed = t.elapsed;
if(key_pressed(KEY_SPACE) && interruptable)
{
return true;
}
}
noise.volume = end_volume;
return false;
}

We will use a sound object to play our in-game music, so let us declare this object now:

sound music; // The one sound object used for all music

Next, we need a variable to keep track of the current volume of the music as the player configured it using the page up and page
down commands:

int music_volume; // Volume for all game music

Now let us define a function to change the currently playing music. This function will accept as parameter the filename of the
music to change to. The elegant thing about this function is that it will fade out the currently playing song, then fade in the new
one, resulting in a graceful transition from one track to the next.

// Changes the music, fading if appropriate
void music_change(string music_file)
{
if(music.playing)
{
fade(@music, music_volume, -50, 500, false);
}
music.stream(music_file);
music.volume = -50;
music.play_looped();
fade(@music, -50, music_volume, 500, false);
}

And while we are at it, let's define some more music-related functions:

// Fades out the music
void music_stop()
{
if(music.playing)
{

fade(@music, music_volume, -50, 500, false);
}
}

// Pauses the music abruptly
void music_pause()
{
music.pause();
}

// Resumes the music abruptly
void music_resume()
{
music.play_looped();
}

// Increases the volume of the music gracefully
void music_volume_up()
{
if(music_volume <= -5)
{
int old_volume = music_volume;
music_volume += 5;
fade(@music, old_volume, music_volume, 200, false);
}
}

// Decreases the volume of the music gracefully
void music_volume_down()
{
if(music_volume >= -55)
{
int old_volume = music_volume;
music_volume -= 5;
fade(@music, old_volume, music_volume, 200, false);
}
}

Note that in the remainder of our source code, we will never manipulate the music object directly, but only through the functions
we just defined. Software engineering calls this approach modularization. It makes sure that whenever something is wrong with
the music, we need only check the correctness of our music functions, because since all music-related activity takes place
through these functions, this is where any music-related bug would logically be found.

Another advantage of modularization is that if the underlying implementation of the music module changes in the future, no source
code outside this module would need to be adjusted because the rest of our source code does not rely on the underlying
implementation but only upon the functions which our module exposes to the outside world. For example, a future incarnation of
our music module might use midi files and sound fonts instead of prerecorded audio. All we need to change in order to
accomplish this would be the functions we just defined. No other part of our source code would need to be modified because all
music-related activity is channeled through the above functions.

6. Sound effects

In part II of this tutorial series you learned about the sound_pool class. Let us use this class now to define some sound-related
functions:

sound_pool pool; // The sound pool used for all sound effects

// Plays a sound in the center of the stereo field
void sound_play_centered(string filename)
{
pool.play_stationary(filename, false); // Not looping
}

// Pause all sounds
void sound_pause_all()
{
pool.pause_all();
}

// Resume all sounds
void sound_resume_all()
{
pool.resume_all();
}

// Gets rid of all sounds at once
void sound_purge()
{
pool.destroy_all();
}

Exercise

Our example game is relatively simple in that it uses only centered sounds. As your games become more complex and
challenging, you will want to use positional audio. Extend the sound module accordingly. Apply what you know about the
sound_pool class, and consult the reference when in doubt.

7. Speech

The third and final auditory component of our game will be spoken messages. For now, the voice which delivers them will be
computer-generated, so we will use the tts_voice class you already saw in the first two installments of Practical BGT.

tts_voice voice; // The voice used for all in-game messages

// Speaks some text
void voice_speak(string text)
{
voice.speak(text);
}

// Speaks some text, interrupting text which is being spoken
void voice_speak_immediately(string text)
{
voice.speak_interrupt(text);
}

// Silences the voice
void voice_stop()
{
voice.stop();
}

8. Sound design

Exercise

From the specification in section 4, extract a list of required sounds.

Here is my solution:

step.wav: A standard footstep
step_alert.wav: A footstep after which the available directions have changed
item.wav: The player has reached an object

9. Inventory

Now we are slowly making our way from the very general-purpose modules to the more game-specific functionality. In this
section we deal with the player's inventory list, a dynamically expanding or shrinking list of all the items the player is currently
carrying. We are going to use an array to keep track of the handles to the items in the player's inventory, and we will provide
functions through which this array is manipulated.

item@[] inventory; // The inventory is an array of handles to items
int inventory_position = -1; // Current position in the inventory list

// Adds an entry to the inventory list
void inventory_add(item@ entry)
{
int old_length = inventory.length();
inventory.resize(old_length+1);
@inventory[old_length] = entry;
}

// Removes an item from the inventory; returns true on success, false on failure
bool inventory_remove(item@ entry)
{
uint index; // For searching the inventory for the entry
for(index=0; index<inventory.length(); index++)
{
if(inventory[index] is entry) // Found it
{
break;
}
}
if(index >= inventory.length()) // Entry not found
{
return false;
}
// Move all above entries one slot down
for(uint i = index+1; i<inventory.length(); i++)
{
@inventory[i-1] = @inventory[i];
}
// Finally, make the inventory one slot smaller
inventory.resize(inventory.length() - 1);
inventory_position = -1; // Make it invalid
return true;
}

// Gets the current inventory entry
item@ inventory_current()
{
if(inventory_position < 0) // No current item {
return null;
}
return @inventory[inventory_position];
}

void inventory_cycle()
{
if(inventory_position+1 >= int(inventory.length()))
{
inventory_position = 0; // Go back to first if at the end
}
inventory_position++;
}

10. Items

The inventory list is all about managing and cycling through the items in the player's possession. Let us now turn to the question of
what an item is in the first place.

In an adventure game, an item can be defined as any object with which the player might want to interact. For example, if one of
the game's challenges consisted of unlocking a door with a key, both the door and key would be considered items. We use the
word item here as a generic term for any object in the game world, including treasures, monsters, doors, and weapons.

For our purposes, an item must provide at least three pieces of information:

1.

2.
3.

A name; this is how the game refers to the item when describing what the player sees, or when cycling through the
inventory.
A description; this is what the player hears when examining the item.
A takeable flag; this determines whether or not the item can be picked up. Note that flag is just programmer lingo for bool.
Setting a flag to true is sometimes referred to as just setting it, and setting it to false can also be called clearing it.

Let us translate this into code:

class item
{
string name;
string description;
bool takeable;
item()
{
alert("Programming Error", "Cannot make item without parameters.");
}
item(string name, string description, bool takeable)
{
this.name = name;
this.description = description;
this.takeable = takeable;
}
}

11. Areas

Areas comprise the geography of an adventure game. They have been known under various names, including rooms, levels, or
maps. But no matter how we refer to them, they are the world through which the player navigates.

Some mainstream games go out of their way to simulate real world geography, complete with heights, textures, and weather
conditions. While all of this is certainly possible in BGT, for simplicity's sake we shall stick with a basic setup here, keeping track
of just the locations of walls and items on our maps. We represent this information in two arrays, both of them two-dimensional.
In addition, every area will provide a general description of where the player is located. This will be announced when the player
chooses to look around by pressing the l command.

class area
{
int size_x; // East-west size of area
int size_y; // North-south size of area
bool[][] walls; // true for wall tiles, false for floor space
item@[][] items; // Items on the map
string description; // Announced when looking around
area()
{
alert("Programming Error", "Cannot construct area without parameters.");
}
area(int size_x, int size_y, string description)
{
this.size_x = size_x;
this.size_y = size_y;
this.description = description;
walls.resize(size_x);
items.resize(size_x);
for(int i=0; i<size_x; i++)
{
walls[i].resize(size_y);
items[i].resize(size_y);
for(int j=0; j<size_y; j++)
{
walls[i][j] = false;
@items[i][j] = null;
}
}
}
void add_wall(int x, int y)
{
walls[x][y] = true;
}
void add_item(item@ entry, int x, int y)
{
@items[x][y] = @entry;
}
bool get_wall(int x, int y)
{
if(x<0 or x>=size_x or y<0 or y>=size_y)
{
return true; // The wall of the universe
}
bool ret = walls[x][y];
return ret;
}

item@ get_item(int x, int y)
{
item@ ret = @items[x][y];
return ret;
}
}

12. Player position

To represent the player's current position, we need to store the current area as well as the player's current x and y coordinates.

area@ player_area;
int player_x;
int player_y;

13. Looking around

Let us now define a function which looks out from the player's location in a given direction, and reports on what the player sees.
This function takes an x and a y increment. For instance, to look eastward, you would supply an x increment of 1 and a y
increment of 0. To look south, you would supply an x increment of 0 and a y increment of -1.

void look(int x_increment, int y_increment)
{
int x = player_x;
int y = player_y;
string direction; // Name of direction, such as east or north
if(x_increment<0) // Westerly
{
if(y_increment<0) // Southwest
{
direction = "southwest";
}
else if(y_increment==0) // West
{
direction = "west";
}
else // Northwest
{
direction = "northwest";
}
}
else if(x_increment==0) // North/south
{
if(y_increment<0) // South
{
direction = "south";
}
else if(y_increment==0) // Right here
{
direction = "here";
}
else // North
{
direction = "north";
}

}
else // Easterly
{
if(y_increment<0) // Southeast
{
direction = "southeast";
}
else if(y_increment==0) // East
{
direction = "east";
}
else // Northeast
{
direction = "northeast";
}
}
while(true)
{
x += x_increment;
y += y_increment;
if(player_area.get_wall(x, y))
{
break; // Hit a wall
}
if(player_area.get_item(x, y) is null)
{
continue; // Nothing there, so look on
}
// Found an item
int distance = absolute(x-player_x) + absolute(y-player_y);
voice_speak(player_area.get_item(x, y).description + ", " + distance + " " + direction);
}
}

14. Moving around

Exercise

Write a function
bool move(int x_increment, int y_increment)
that moves the player in the given direction if possible. The function should return true if the movement was successful, false
otherwise. If this function returns false, the player's position should not be altered. This function should play the appropriate
footstep sound.

15. Object manipulation

Exercise

Write a function
examine()
that examines the item at the player's position. The function should speak the description of the item, or an appropriate message if
there is no item at this position.
Hint: To find out if the handle x indeed refers to an object, use the expression (x !is null).

Exercise

Write a function
use()
that uses the object at the player's position, giving an appropriate spoken message if there is no object there, or if the object at
this position cannot be used.
Hint: You will modify this function frequently as you add items to your game.

Exercise

Write a function
use_inventory()
that uses the currently selected object in the player's inventory, giving an appropriate spoken message if no object was selected,
or if the currently selected object cannot be used.
Hint: You will modify this function frequently as you add items to your game.

16. Game loop

Exercise

Write a function
game_loop()
that continually checks for keypresses and carries out the appropriate commands until the game is over.
Hint 1: Use a flag for the game over condition, and set this flag at an appropriate place, for instance in your use or use_inventory
function.
Hint 2: You may wish to write a helper function for pause mode.

17. Use your imagination

Exercise

Using the adventure game framework you created, begin adding areas and items to your game.
Hint: It might be a good idea to create a separate initializer function for each area.

Packaging Files in Your Executable

It is possible to include a BGT pack file inside a compiled game executable. This means that you can ship an arbitrary number of
files that you do not wish to store separately on disk, as part of the program itself. In order to do this, follow the steps below:

1. Generate a pack file using the pack_file object in the engine. In this pack you store all the files that you wish to include in the
final executable.

2. In your script code, write:

#include "packname.dat"

Replace packname.dat with the name of the pack that you created in step 1. The engine automatically detects whether a given
include file is a pack, or a regular script file. Note that you may not include more than one pack file in each BGT game.

3. When it is time to access the pack file, simply specify * as the pack file name wherever the name of the original pack file
would normally have been given. For example, at the start of the main function you might write:

set_sound_storage("*");

If you run this script from source, the sounds will be searched for in the pack file on disk just as if you had specified the name
that you gave in the include statement. However, when this script is compiled into an executable it will refer to the contents of the
pack file that is stored inside the program itself.

You may access the contents of the pack directly using the pack_file object, as follows:

pack_file pack;
pack.open("*");

The same rule as above applies regarding execution of the script from source as opposed to when it is being run as an
executable. In short, whenever a script is run from source the pack file that was included will be accessed directly on disk.

4. You may now compile your script, and the given pack file will be included along with your compiled game code in the resulting
executable. As mentioned above, * should be used wherever the real pack file name would have been specified if you wish to
access the pack inside the program.

Pathfinder Tutorial

Contents

1. Introduction
2. What does the pathfinder do?
3. Specifying the size of the map
4. Setting a callback function
5. Putting it all together
6. Exercises

1. Introduction

Mazes, or maps, abound in games, and this would include audiogames as well. If you have played a sidescroller before, you will
have navigated a maze with the four primary directions of right, left, up, and down. A first person shooter, on the other hand, will
usually present you with a type of map in which up and down are not so relevant; instead, you usually walk forward with the up
arrow key and turn around with the left and right arrow keys.

Both cases are examples of two-dimensional maps, in which positions are given by two coordinates, usually called x and y. In the
case of a sidescroller, increasing the value of x might mean moving right, and increasing the value of y might mean moving up. We
say that the x axis points right and the y axis points upward. In the first person shooter we might have the x axis pointing east and
the y axis pointing north.

Part of the challenge of most map-oriented games is navigation. The player listens for the location of an object, turns in that
direction, walks forward, finds a dead end, backtracks, tries another route, and so on. As game developers we sometimes wish
to bestow the same kind of intelligence, or appearance of it, upon entities other than the player. We want our enemy robots to
chase the player, skillfully circumventing acid pools as they do so. We want our peasants to go about their daily business without
bumping into walls. And as the game world changes, we want its inhabitants to be smart enough to correct their course.

2. What does the pathfinder do?

Simply put, the pathfinder object finds a path from a source point to a destination point on your game map. The path is
represented as an array of vectors, in other words, a list of points.

3. Specifying the size of the map

Before the pathfinder can even begin its work, it needs to know the size of the game map. You specify this by calling the
create_map method, as in the following code:

void main()
{
pathfinder holmes;
holmes.create_map(10, 5);
// More code here.
}

This would tell the pathfinder that your game map has a width of 10 and a height of 5 squares. In other words, your x
coordinates will be in the range from 0 to 9, while your y coordinates will be in the range from 0 to 4.

4. Setting a callback function

If you are new to BGT, or indeed new to programming, you may not have seen the concept of a callback function before, so a

brief explanation is in order.

Imagine the following situation: you are still at the office, and you know that upon arriving home you have only twenty minutes to
catch a train, so you call home to ask a family member to pack a suitcase for you. Five minutes later your family member calls
you back to ask whether you wish to take the black tie or the red tie with you, to which you respond that you would like the red
one. Another ten minutes later she calls you back once again to inquire if she should prepare a meal before you leave, to which
you respond that you probably don't have enough time. Another twenty minutes later she calls you back yet again to ask how
long you will be away.

Let us now translate this into programming terminology. Asking a family member to pack your suitcase means calling a method
called pack_my_suitcase on an object of class family_member, like so:

void main()
{
family_member matilda;
matilda.pack_my_suitcase(); // Impolite but to the point.
}

Now, what if Matilda needs more information to fulfill your request? Let us define a function to take care of her questions:

string my_callback(string inquiry)
{
if(inquiry == "Tie color?")
{
return "Red.";
}
else if(inquiry == "Prepare meal?")
{
return "Nope.";
}
else if(inquiry == "Time away?")
{
return "Three days.";
}
else // We don't understand the question.
{
return "Beats me.";
}
}

Finally, let's modify our main function to tell Matilda about our callback function so she can call it if necessary:

void main()
{
family_member matilda;
matilda.set_callback_function(my_callback);
matilda.pack_my_suitcase(); // Impolite but to the point.
}

Just like in our real-life scenario above, Matilda will call our my_callback function repeatedly until she has gathered all required
information. Note that we never call my_callback directly; we simply provide it in case Matilda might need to call it. Providing a
callback function is indeed very much like leaving a phone number in case there are any questions. We never call our phone
number directly, but it gets called by another entity if that other entity deems it necessary.

Let us now apply what we just learned to the pathfinder object. Up until now we have told it only the size of our map. If we now
ask it to find a path between two locations, it cannot perform this operation without asking us a lot of questions because it

doesn't know anything about our map except its size; in particular, it has no idea where walls are blocking the path, doors might
need to be opened first, or acid pools might make travel hazardous if not impossible. In short, we need to provide a callback
function which tells the pathfinder how difficult it is to get from one square to an adjacent square. The pathfinder will always
decide on the least difficult path, adding up the difficulties along the way.

Let us look at how such a callback function might be defined:

int maze_callback(int x, int y, int parent_x, int parent_y, string user_data)
{
if(maze[x][y]==1)
{
return 10;
}
return 0;
}

The pathfinder calls this function when it would like to know how difficult it is to get to the square at coordinates x and y. In
addition, the pathfinder will also provide the coordinates of the square it is coming from, called parent_x and parent_y. This is
useful when some squares are easier to reach from one direction than another. A one-way door, for instance, might be very easy
to pass through from left to right, but impossible to pass through from right to left. A brick, for example, would find it very easy
to go down while having a hard time going up. Finally, our callback function receives a string called user_data, and we will see
shortly where this is coming from.

What does our callback actually return? We hinted at this a few paragraphs ago by saying that it returns the difficulty of getting to
one square from an adjacent square. This difficulty, otherwise known as cost, or risk, is simply an int in the range 0 to 10. A
value of 0 means there is absolutely no risk or cost associated, while a value of 10 means it is impossible to go there, just as
impossible as walking through a wall unless you happen to be a ghost, or possibly Chuck Norris. So 0 means free right of
passage, while 10 means no way. Of course, some paths are neither completely clear nor impossible to take, and these will have
cost values greater than 0 but less than 10. For example, you might assign a value of 2 to shallow water, a value of 6 to deep
water, and a value of 9 to fire.

In our running example, the callback function simply consults an array called maze to see if there is a wall at the location in
question. If so, it returns 10, indicating that it is impossible to go there. Otherwise it returns 0, indicating the way is clear.

Of course we need to tell the pathfinder about our callback function. The following code demonstrates this:

void main()
{
pathfinder holmes;
holmes.create_map(10, 5);
holmes.set_callback_function(maze_callback);
// More code here.
holmes.destroy_map();
// Maybe even more code here which doesn't need pathfinding.
}

As you can see, we tell the pathfinder about our callback by calling its set_callback_function method. This method takes a single
parameter, which is the callback function itself, in this case, maze_callback. If you are curious how we managed to pass a
function to another function, consult the language tutorial about the funcdef keyword. Note that this is entirely optional; you don't
need to know how it works in order to use it. Such are the joys of well-designed programming languages.

In the code above, we seized the opportunity to introduce yet another method, called destroy_map. This method will free some
memory that the pathfinder uses internally to calculate paths, and so it is recommended you call destroy_map when you won't
need the pathfinder for some time, and then call create_map once more if you need it again. For the technically interested, this
memory is also freed as soon as the pathfinder object is destroyed. It is, however, entirely safe to ignore the previous sentence if
it doesn't make sense to you. Object creation and destruction are covered in great detail in the language tutorial.

5. Putting it all together

Now that we covered the infrastructure, let's do some real pathfinding! It is time you studied the source code example given in
the pathfinder chapter of the BGT reference. You might want to copy it to a file on your hard drive in order to experiment with it,
and to be able to have it open in another window while you continue reading.

You might begin by mindfully reading the code top to bottom to see which components you recognize. Can you find where the
pathfinder object is created, and the map size is defined? Can you spot the callback function, where it is defined, and where it is
passed to the pathfinder object?

The program is well-commented so you should have no trouble figuring out what it does. You might call it a maze solver. It lets
you walk around a grid using the arrow keys, placing and removing walls along the way. Finally, you can define a starting
location, move to the destination location, and let the solver do its magic. What you will hear is a list of directions from start to
destination. Now, I wish I would have had something like this when I was playing Zork.

Let us look at the statement that actually asks the pathfinder to find a path for us:

vector[] path=holmes.find(start_x, start_y, x, y, "");

The find method takes as arguments the starting coordinates, the destination coordinates, and finally a string called user_data.
This string is not used in the example, and so it is left empty. As it happens, this is the exact string which gets passed to the
user_data parameter of your callback function. This might be useful when you want the callback function to behave differently
depending on the circumstances, or indeed depending on the reason why a path needs to be found.

The find method returns an array of vectors. For the purposes of the pathfinder, a vector is simply a pair of x and y coordinates,
i.e. a point on your map. So the elements of the array make up the route of travel from starting point to destination. If no path
could be found, the find method will return an empty array.

6. Exercises

1.

2.

3.

4.
5.
6.

7.

Modify the example such that the user can place acid pools at arbitrary locations on the map. An acid pool is not entirely
impossible to walk through, but since it is quite an unpleasant experience the solver should minimize contact with them.
Now let's add another kind of hazard. Modify the example such that the user can place puddles of water at arbitrary
locations on the map. Such puddles are much less inconvenient than acid pools, but the solver should still avoid them if
possible, except, of course, if this involves walking through acid instead.
Can you modify the callback function such that moving east is more difficult than moving west, and moving north is more
difficult than moving south?
Can you make the feature of exercise 3 optional, but still use only one callback function?
Make a ghost which can pass through walls, but only when going east.
In a first-person shooter, how might you code an enemy who chases the player? How can you ensure the enemy is smart
enough to correct its course when the player moves?
Look up the desperation_factor property of the pathfinder object in the reference. Can you see how this would be useful?
Consider the example of enemies becoming more desperate when the player approaches the end of the level. What other
scenarios can you think of?

Performance Optimizations

Introduction

You have a game that you are relatively happy with. It has nice sounds, seems stable, and is enjoyable. However, you start to
notice that as the game grows, it is becoming slower and slower. Is all hope now lost? Not quite.

There are plenty of steps you can take to make your script run optimally. Below is a summary of the most effective techniques
which should make you notice a difference immediately. They are ordered roughly after how useful they are in a general gaming
context.

Optimize sounds with preloads

In larger games, you most likely have a lot of sounds that are playing simultaneously. Often times you will probably be using the
sound pool to manage your audio environments. The sound pool loads all the sounds that you ask it to play, into memory in their
entirety. It does not stream them as this would create a lot of problems when the number of playing sounds grew. However,
loading sounds from disk is very costly in terms of CPU cycles, and as a result it has a great impact on the general performance
of the game. To combat this problem, BGT has a very useful but often overlooked feature. Whenever you load the same sound
more than once, it does not actually read the sound from disk again. It simply tells the new sound object to refer to the same
internal memory location as the existing one does where the actual data is stored, while still giving you a completely independent
sound to work with in terms of pan and volume etc. This is called cloning. The benefits of cloning become especially apparent for
encrypted Ogg Vorbis files as the engine does not only have to decrypt the data after reading it from disk, but also decode it
which takes an additional amount of time that may be significant if the sound is more than a few seconds in length. If this takes
place in the middle of gameplay where a lot of things are happening, you will notice a severe lag as well as possible stuttering of
the sounds that are already playing. As mentioned earlier, the sound pool loads all the sounds from disk whenever you ask it to
play them. Thus, you will run into this performance bottleneck very quickly with the sound pool as well as with any other code
that loads sounds from disk frequently.

Luckily, there is a very easy solution to this problem. It is to simply load all the sounds into memory at the start of each level.
When this is done and what sounds are loaded will naturally depend on the characteristics of your game, but the idea is to load
any sound that you might need during the game before the actual gameplay starts. You keep these sounds around constantly,
usually in a global array, so that they never go out of scope and are thus never unloaded until you specifically want them to be.
When the sounds are loaded, the BGT engine is able to clone all of them when the sound pool or any other piece of code
attempts to load them. The result is that while you will introduce a more or less noticeable lag while the sounds are initially
loaded, you will also see that performance increases dramatically during the gameplay as nothing needs to be loaded from disk.
Disk reading and writing is one of the slowest processes in an operating system, and Windows is no exception. Add decryption
and Ogg Vorbis decoding on top of this, and you will see that the benefits of preloading sounds really cannot be stressed enough.
Below is a basic example of how the preloading mechanism can be constructed:

// Create a small class that will hold the sound handle for each preload along with its filename.

class preload
{
sound@ handle;
string name;
}

// Declare a global array of preload class instances.

preload[] preloads;

// The following function takes a filename as its only argument and loads the sound as a preload.
// It returns true on success and false on failure.

// It will fail if the sound has already been loaded as a preload, or if something goes wrong while actually loading it.

bool preload_add(string filename)
{

// Verify that the sound hasn't already been loaded as a preload by comparing the filenames.

for(int i=0;i<preloads.length();i++)
{
if(preloads[i].name==filename)
return false;
}

// Load the preload, and return true if everything went well.

sound temp;
temp.load(filename);
if(temp.active==false)
{
return false;
}
preload loader;
@loader.handle=temp;
loader.name=filename;
preloads.insert_last(loader);
return true;
}

// The following function resets all of the preloads.

void reset_preloads()
{
preloads.resize(0);

// Call the garbage_collect function to ensure that all data is cleaned up immediately.
// Comment out this call if you don't want this extra delay here.

garbage_collect();
}

If you use this code, you simply have to call the preload_add function multiple times to load all of your sounds when appropriate.
reset_preloads is a convenience function that not only resets the global array, but also ensures that the data is actually unloaded
from memory by running a full garbage collection cycle. If you use the sound pool, it is a good idea to call the destroy_all method
in it before unloading the preloads so that the garbage collector can do its work more effectively. Note that a full garbage
collection cycle might take some time depending upon the behavior of your script, so you may want to test how severe this is and
then determine if it is appropriate to do at this point in time. It is generally a good idea to invoke the garbage collector at the end
of a game round, though.

Call wait often

Knowing when to use the wait function is absolutely crucial for the performance of your game. When you call wait, you give the
computer a chance to do other things. Windows is a multitasking operating system, which means that all the programs that are
running at any given time share the resources of the hardware. wait should be called, usually with a parameter of 5, in any loop
that runs for an extended period of time such as the game's main loop. Failing to do this will result in a CPU usage of 100% for
the game process. This in turn will cause general instability of the operating system in many cases, and will make most laptop fans

go haywire.

wait also performs other useful tasks such as keeping the game window's internal event queue up to date, and invoking the
garbage collector in small increments. This means that if you fail to call wait in a script loop that generates garbage, which is to
say creates and destroys some resource in each iteration, the garbage collector will be full of data without getting a sporting
chance to clean it up effectively. BGT has a safety mechanism which kicks in if the garbage collector is close to overflowing, but
this forces the engine to run a full collection cycle which is generally not what you want during gameplay. In short, wait not only
gives the CPU a rest; it also keeps the engine running smoothly. Use it!

Avoid streams when possible

Streams are very tempting to use because they start playing the sound very quickly, and there is not a great amount of memory
overhead for them if compared to loaded sounds. However, streams should be used with care. While certainly tempting, they
have one serious flaw. A stream achieves its quick starting time by only loading a little piece of the data into memory and then
beginning to play it. While this first piece is playing, the stream is busy loading the next little chunk. As soon as the first piece has
finished, the second one begins playing and the third is then loaded and prepared for playback. This sounds simple, but can
cause a lot of issues when it comes to performance if too many streams are playing at once. If you have an Ogg Vorbis sound
that is encrypted, the engine needs to perform the following steps for each and every individual chunk of data:
1. Read it from disk into memory. Disk reading and writing is, as stated above, one of the slowest processes in an operating
system (Windows included).
2. Decrypt it. This is generally pretty fast, but it still adds to the overall processing time quite a bit when done many times.
3. Decode the decrypted data from Ogg Vorbis back to raw audio samples. This also takes a bit of time as the Ogg Vorbis
format is quite complex.
4. Finally feed the decoded data to the audio device which involves copying more memory.

All of these things combined result in a fairly costly operation as far as CPU cycles are concerned. It generally works fine when
you have just a few streams going, but if this number grows you will start to notice high CPU usage as well as occasional
stuttering of other sounds or even little repeating chunks in those streams that may not get enough processing time to load their
next chunk before the previous one has finished. Therefore, you only want to use streams for sounds that are very long and that
do not play right in the middle of a fast, action packed game level. As a rule of thumb, it is better to use more memory rather than
more processing power. Load sounds as much as possible and take advantage of the cloning feature outlined above. Streams
have their place, but consider what happens in the background before deciding to use them.

Use the reserve method in the array

If you have an array to which you know you'll be adding and removing a lot of elements using insert_last, insert_at and friends, it
is a good idea to reserve memory in advance. Each time an insert method is called, it will allocate just enough memory to store
the new item. Similarly, each time a remove method is called it will free the memory for the given item directly. This may seem
like a good thing, but can actually hurt performance a great deal because memory allocation takes time. Therefore, allocating
more memory than you actually need is a common optimization strategy in order to avoid many small allocations along the way.
The reserve method will allocate memory for the requested number of items internally, without actually resizing the array. In other
words it is possible to have an array with 0 elements in it, but plenty of memory reserved for future use. This means that the next
time you call insert_last for instance, it will not allocate any new memory but will simply place the newly created item in the
already reserved memory.

The following script will show you the difference between inserting one item at a time without reserving memory, versus reserving
memory and then inserting items:

void main()
{
timer counter;
int[] list;
int[] list2;
for(int i=0;i<100000;i++)
list.insert_last(i);

alert("First result", "Time elapsed without reserved memory: " + counter.elapsed + " milliseconds.");
list.resize(0);
counter.restart();
list2.reserve(100000);
for(int i=0;i<100000;i++)
list2.insert_last(i);
alert("Second result", "Time elapsed with reserved memory: " + counter.elapsed + " milliseconds.");
}

On my machine, the results are as follows:

Time elapsed without reserved memory: 2012 milliseconds.

Time elapsed with reserved memory: 12 milliseconds.

In short, an optimization well worth your time!

Perform as few string comparisons as possible

Comparing a couple of strings may seem like a trivial operation, but if done extensively you will certainly be using more
processing power than you need to. Consider the following script:

void main()
{

// Set up a timer to measure performance.

timer counter;
counter.restart();

string some_data="hello";

for(int i=0;i<100000;i++)
{
if(some_data=="ball")
{
// Do something.
}
}
alert("Result", "The operation took " + counter.elapsed + " milliseconds.");
}

On my machine, this takes an average of about 40 milliseconds. It may be argued that this is not a representative test case for
real world usage, and certainly the test itself isn't but the comparison of strings can easily reach this quantity in a large game if
used extensively. Now, consider the following changed code:

void main()
{

// Set up a timer to measure performance.

timer counter;
counter.restart();

int some_data=5;

for(int i=0;i<100000;i++)
{
if(some_data==5)
{
// Do something.
}
}
alert("Result", "The operation took " + counter.elapsed + " milliseconds.");
}

This takes about 6 milliseconds on average on my system. A big difference, in other words. This goes to show that while both
operations seem trivial, the latter is significantly faster than the former. It can thus be concluded that, if at all possible, use integers
rather than strings at all times. If defining enemy types, for instance, don't use strings such as "robot", "alien" or "monster", use
enums or constant integers instead. While perhaps not immediately noticeable, the performance difference will benefit you in the
long run as your code expands.

Clean up your garbage

The usefulness of invoking the garbage collector explicitly was briefly outlined in the section dealing with cloned sounds, but it
does not hurt to reiterate this one more time. If you have a point in your game where latency is not a problem such as during a
game over sequence or before the main menu is started after a game round, it is a very good idea to call the garbage_collect
function explicitly to clean up all the memory that has not yet been freed by the incremental garbage collection process. This is
not only beneficial in terms of memory usage, but also in terms of performance because the incremental garbage collection
processor has less work to do. For more information on what the garbage collector does and why it is important to be aware of
it, see the help topic for the garbage_collect function in the reference.

Conclusion

Performance is always a relative thing. You can be as brief, or as thorough as you like. The above tips are meant to serve as
some general guidance to avoid common mistakes and pitfalls that you often encounter during development. The gameplay is the
important thing, but do spend time on performance. Your users will thank you for it and so will your own processor.

Registration in BGT

1. Introduction

A lot of users are requesting a way to protect their games from piracy, which is a very understandable attitude considering how
much time and money that is usually spent on a commercial game. However, the sad truth is that there is no bulletproof way of
accomplishing this. If you search for the name of your favorite software plus the word crack on Google or any other popular
search engine, you will almost always find someone who has taken the time to break the registration system and post either a
patch, a key generator or some other way of bypassing the security system. The problem becomes even greater when using a
game engine, because the way in which things like unique computer ID's are generated will be the same for every game produced
with it. This is not to say that the ID itself will be the same for each game, but the internal generation code will ultimately be the
same. In other words, the more BGT does for you behind the scenes in order to simplify your work, the easier it will be for a
cracker to break almost every game produced with the engine. For this reason, BGT provides an absolute minimal set of features
for registration management and it is up to you as the script writer to think of a way of generating keys etc. The advantage with
this is that every game will do it a little differently, making it a lot harder for someone to generate some ultimate BGT game crack
that will work everywhere. This is still far from a guarantee that your game will not be cracked, but it does make it harder as the
cracker will have to start from scratch for each new game.

2. Methods

So how do I go about protecting my product, you ask? There are three main types of registration systems used today. Below
follows a list of them, with pros and cons of each:

2.1. Name/key

This system is by far the simplest, most user friendly, and least secure. The user simply enters their name, pays for the software,
and gets a key back that is a scrambled version of their name plus some unique constant identifying the game in question. Without
this game specific constant, a user who purchases one game from you could use the same key to register all your other games as
well. The only difficult part with this system is to scramble the user's name in a creative way. There is no right or wrong way of
doing this, and there is really only one rule. Make it as irrecognizable as possible and you are off to a good start. BGT provides a
large number of string related functions so it's really up to your imagination which ones you want to use and how to combine
them.

Very user friendly. A customer who has purchased the game can back up their name and key and reuse this information if
they get a new computer for example, since it never changes.
Easy to implement. all that is required is a creative way of obscuring the user's name, a constant identifying your game
which is scrambled in with the name to form the final key, and you're done.

Pros:

Cons:

Trivial to crack. All a user has to do in order to give your game to their best friend and the best friend's relatives is to hand
out their name and key.

2.2. Product ID/key

This is very similar to the name/key system mentioned above, with one major difference. Instead of basing the unlock key on the
user's name, you base it on some unique hardware information on the user's computer. You will want to combine the generated

unique ID with a game specific constant as described in the section above, or the product ID will be the same for all of your
games on a given machine. You must provide a way for the user to obtain this product ID from within your game, and the user is
then required to submit this information along with their payment. Then, the steps are the same as with the name/key system,
where you have to scramble the user's product ID in as creative a way as you can think of in order to generate the key. You
probably do not need to use the game specific constant while creating the key, however, as this should already have been used
while generating the product ID to make the ID different for each of your games. The main issue with this system is that you will
need to provide new keys to a user as soon as he or she makes some change to their hardware. BGT provides a function to
generate a unique computer ID based on several different hardware components, however this ID will most certainly change if
the user reformats their PC. This makes for a lot of maintenance work, and plenty of frustration for users who have to wait for
you to generate keys for them.

Considerably more secure than name/key. A user cannot simply copy over their registration key to another machine.
Easy to implement. This requires about the same amount of work to implement as name/key, given the unique computer
ID generation function provided in BGT.

Pros:

Cons:

Requires a lot more maintenance. As soon as the user reformats their computer or changes a certain hardware
component, they will need to contact you in order to get a new key just to keep using the product they have already paid
for.
Vulnerable on a massive scale. If the product ID generation system in BGT is cracked, every game that uses it will be
cracked in an instant.

2.3. Online based

This can be considered a sort of hybrid between the above mentioned systems. The basic idea is to have a central server hosted
somewhere that the game connects to when registering, and the server then tells the program whether the registration is accepted
or not. It is common to use the name/key system together with the online server, as that allows a user to register their game on as
many computers as the server allows without having to request different keys for each machine which would be the case if a
product ID was used. Thus the main advantage of this system is that the user can reuse the same name and key, but only as many
times as the developer allows for. However, as with each system described here, this one also has a number of drawbacks. First,
the user needs to be connected to the Internet at the time of registration. This is usually not a problem for most customers, but
may be in rare cases. Second, the server needs to be online at all times in order for registrations to work at all. Third, after the
registration has been acknowledged and accepted by the server the game still needs to remember the fact that it is registered as
to avoid having to connect to the server each and every time the program is started. This means that there must be some local
information stored that tells the game that it is registered, without using the server. As a result, a cracker can simply simulate this
information and get the game registered without the server having the slightest idea of what happened.

Pros:

Cons:

Full control. You as the developer have full control over who is registered, how many computers they are using the game
on, and can limit the number of allowed registrations at any time.
Reusability. Since the customer does not need a product ID but can use their name instead, they are able to register it on
multiple computers with the same information provided that you allow for this.

Requires an Internet connection at the time of registration. As stated previously this is usually not an issue, but it may be. .
Completely dependent on the server. If the server goes down for any reason, the paying customers will not be able to
register and use the product they have purchased.

Harder to implement. A server script is needed for the system to work, which requires a good understanding of a scripting
language such as PHP as well as an excellent awareness of security design techniques.
Vulnerable. Local information must be stored to tell the game that it is already registered. This information is not difficult
for a cracker to forge.

3. Conclusion

After looking at all three of these registration systems, we can come to one obvious conclusion. No matter what we do, our
game will be cracked eventually. If it is a good game and someone feels it is worth the time spent, you do not stand a chance no
matter what measures you have taken. This applies especially to the product ID system described in section 2.2, as the internal
generation code will be the same for all BGT games even though the resulting ID is not the same. On top of this, the product ID
system requires a lot of maintenance and is likely to get frustrating both for the developer and the end users. It is almost inevitable
that the product ID generation code in BGT will eventually be cracked, as are all similar systems if the cracker is determined
enough. When this happens, you have not only got your game pirated; you have also made it a lot harder for the honest
customers to actually use your software. In the end, the product ID has not served any other purpose than to complicate life both
for you and your users while not preventing cracks. The same is true for the online system, as this also has several drawbacks as
listed above. Thus, as an experienced software developer I personally conclude that the name/key system is to prefer, for three
reasons:

1. It is very easy; both on the developer in terms of maintenance and the end users as regarding portability.
2. As soon as the other two systems are cracked which will happen sooner or later, they only complicate things for no purpose.
3. The people who crack your game would not have purchased it in the first place. However, you will get a lot more sales from
honest customers if you show that you trust them with a system that does not limit their use of your software.

Naturally, the choice is yours. I have attempted to outline the three most common types of registration systems in order to help
you make an educated choice as to which one you wish to use. BGT provides the facility required for all three (a large array of
string functions, a computer ID generation function, and access to http servers using either the http object or the url_get and
url_post functions). However please understand that if our product ID generation system is cracked, there is not much that we or
anyone else can do to solve the situation. As per our license agreement, we can not assume any responsibility for loss of profit or
any other direct and/or indirect issues and/or damages that may arise as a result of your use of this software.

Serialization Tutorial

What Is Serialization?

Serialization is the process of taking data inside of an application such as integers, floating point numbers and strings, and
converting them to a stream of bytes. In game related terms, serialization allows you to save game data and load it back again at
a later time. This can be anything from entire levels with enemies and traps, to simple configuration data.

Many developers, when faced with the task of saving game levels, will come up with a very application specific format where the
order in which things are saved and loaded matters greatly, and where it is difficult to adapt the saving and loading routines as
new versions are made. BGT's serialization functionality attempts to circumvent both of these problems by using keys and values.
Each saved piece of information has a corresponding name which is used when it is time to look up the data again when it needs
to be loaded. This means that the order in which things are saved is no longer relevant because each value is looked up based on
its name, rather than based on where in the long list of values it is expected to be stored. Furthermore, when you release a new
version of your game which requires new content to be saved, your old data files will still be compatible with the new version
provided that you are able to set the new variables to default values if they are not found.

Serializing your Data

The first step to serializing data is to decide exactly what needs saving. For instance, a game that only saves its data at the
beginning of each level will not need to store the player's position, since this will generally be set when the level starts. However,
a player's inventory will need to be saved, as this could change from game to game and therefore should not be surmised.

Once you have done this, all your chosen variables need to be stored in one or more dictionaries. Such variables will typically
consist of global variables and class properties.
Dictionaries provide an easy way to assign a name to your variable that can easily be stored, and when inventing your names it is
important to use names that are unique, otherwise values will be overwritten. A simple way to do this is to use the same name as
your variable, including any class names. For example, if you have a map class with an environment stored in a 2d array, you
might store a location in your map as follows:

for(int  x=0;  x<map.length();  x++)
{
for(int  y=0;  y<map[x].length();  y++)
{
map.location[x][y]=0;
my_dictionary.set("map.location["+x+"]["+y+"]",  map.location[x][y]);
}
}

Another  way  might  be 

to 

store 

it 

similar 

to  a  path  value, 

like 

so:

my_dictionary.set("map/location/"+x+"/"+y,  map.location[x][y]);

Either  way, 
saving.
and 

it  needs 

to  be  unique,  and 

recognizable 

to  you  as 

the 

script  writer, 

so  your  values  can  be  consistent  during 

loading

To 
return  a 

serialize  your  data,  you  need 

string, 

that  you  can  easily 

to  call 
save 

the 
to  a 

serialize 

function 
the 

in  BGT,  passing 
serialization 

file.  See 

the  dictionary  as  a  parameter.  The 

function  will 

then

function  chapter 

in 

the 

reference 

for  more  details.

is  perfectly  acceptable 

to 

It 
as  you  have  and 
back 
dictionaries.

serialize  multiple  dictionaries 
the 

strings 

to 
into  one  dictionary  which  could  potentially  cause  problems 

then  concatenating  all 

together 

form  a  whole.  However,  deserializing  will  always 
if  you  have  keys  with 

load 

the  data

in  your  original

same  names 

the 

into  one 

string  by  calling 

the 

serialize 

function  with  as  many  dictionaries

It 

is 

important 

to  note 

that,  while 

the 

resulting 

string 

is  hard 

to 

read,  numbers  are  easily  discernible,  and 

the 

text  data 

is  encoded

in Base64, making it easy to decode the string back into text. This means that if you wish to make your saved data secure to
prevent cheating, you will still need to use the encryption functions provided by BGT.

Deserializing your Data

Deserializing data is slightly more complicated, since you have to make room for an event where a specific value is not saved, for
example in a different version with a new feature that has not yet been used. A simple approach might be to set all variables in a
given class to default values before loading, and then simply overwriting these defaults with all the things that are successfully
retrieved from the data file. You may also want to store a version number, so that in the unfortunate case where there is
absolutely no way to make saved games backwards compatible you can at least inform the user about this and bail out.

To deserialize, you first need to retrieve the data from the storage location, decrypt it if applicable, and call the deserialize
function, passing the string as a parameter. It will return a dictionary object, whose values you will then need to store back in the
relevant variables.

If a value you look for cannot be found in the loaded dictionary, the variable will not be changed, which of course doesn't matter
in this case, as we have assigned a default value to the variable for the game to fall back on.

For an example of how you might go about deserializing your data, see the deserialize function chapter in the reference.

Foundation Layer

The foundation layer consists of functions and objects that are available in the engine itself. These are the bare basics, and are
what all scripts will use.

Examples of foundation objects include timers, used for measuring time accurately, sounds, used for playing and manipulating
audio, and dictionaries, used for storing and retrieving large numbers of variables.

Foundation functions include error handling, variable conversions, keyboard management and simple user interface functions.

Foundation Layer

The foundation layer consists of functions and objects that are available in the engine itself. These are the bare basics, and are
what all scripts will use.

Examples of foundation objects include timers, used for measuring time accurately, sounds, used for playing and manipulating
audio, and dictionaries, used for storing and retrieving large numbers of variables.

Foundation functions include error handling, variable conversions, keyboard management and simple user interface functions.

get_sound_master_volume

This function gets the master volume for the sound mixer.

double get_sound_master_volume()

Parameters:
None.

Return value:
The master volume between -100 and 0. Default is 0.

Remarks:
The master volume works similar to the master fader on a mixing desk. It controls every sound that is played during the game
and works independently of a sound object's volume control. This means that you can continue treating the volume of individual
sound objects exactly the same as before, where 0 is always the maximum based on the current master volume. This can be
useful if your game uses quiet TTS output and has a rather loud or busy soundscape.

Please note that these functions do not apply to the Windows volume control, only sounds that are played from within the game
itself.

Example:
// Set the master volume to -30 and retrieve the new value.

void  main()
{
set_sound_master_volume(-30);
alert("Sound  master  volume",  "The  sound  master  volume  is  "  +  get_sound_master_volume()  +  ".");
}

get_sound_storage

This function gets the location of the pack file in which the engine will look for sound files loaded through the sound object.

string get_sound_storage()

Parameters:
None.

Return value:
The name on success, or an empty string on failure.

Remarks:
Note that both \ and / can be used to specify paths.

If an empty string is returned by this function, it means that the engine will not look for sounds in a pack file but will attempt to
locate requested files on disk instead. This is the default behavior.

If the function returns *, it means that the engine will attempt to look for sounds in a pack file that has been included into the
program itself. For more information on how this is done, see the tutorial on packaging files with your executable.

Example:
// Tell the engine to look for sound files in a file called sounds.dat, and then print this.

void  main()
{
set_sound_storage("sounds.dat");
alert("Sound  storage",  "The  sound  storage  is  "  +  get_sound_storage()  +  ".");
}

list_sound_devices

This function lists all the devices that can be found on the system.

string[] list_sound_devices()

Parameters:
None.

Return value:
An array containing the friendly names of all available devices, or an empty array on failure.

Remarks:
The devices will be listed in the order they appear in the Windows device list.

Example:
// Enumerate all the devices on the system.

void  main()
{
string[]  devices;
devices=list_sound_devices();
if(devices.length()==0)
{
alert("Error",  "No  devices  can  be  found  on  your  system.");
exit();
}
for(uint  i=0;i<devices.length();i++)
{
alert("Device  "  +  (i+1),  devices[i]);
}
}

open_sound_device

This function opens an audio device for playing sounds.

bool open_sound_device(int device_id)

Parameters:
device_id
The ID of the device to be opened.

Return value:
true on success, false on failure.

Remarks:
The device ID is 0-based, and matches the allotted index in the array returned by the list_sound_devices function.

To use a different device to the default, the device must be opened before any call to the load or stream methods in the sound object. If this is not
done, the default device that is selected in the Control Panel will be used.

The engine runs off one master device only, and the device cannot be changed once it has been opened (either by an explicit call to this function or
implicitly when either the stream or load method of any sound object is called for the first time).

Example:
// Play a sound and explicitly specify a device.

void  main()
{
sound  test;
string[]  devices;
bool  success;
devices=list_sound_devices();
if(devices.length()==0)
{
alert("Error",  "No  devices  can  be  found  on  your  system.");
exit();
}
if(devices.length()==1)
{
alert("Information",  "Only  one  device  can  be  found  on  your  system  and  will  now  be  used.");
}
success=open_sound_device(devices.length()-1);
if(!success)
{
alert("Error",  "Device  number  "  +  devices.length()  +  "  could  not  be  opened,  so  reverting  back  to  the  default.");
}
test.load("c:/windows/media/ding.wav");
if(!test.active)
{
alert("Error",  "Cannot  find  one  of  your  windows  sounds.");
exit();
}
test.play_wait();
test.close();
}

set_sound_decryption_key

This function sets the global decryption key that is used to try and decrypt an encrypted sound for playback.

bool set_sound_decryption_key(string key, bool force_encryption)

Parameters:
key
The key that is to be used. This may be either a string or a numeric value.
force_encryption
A boolean (either true or false) that tells the engine whether to allow loading of sounds that are not encrypted.

Return value:
true on success, false on failure.

Remarks:
This function sets the decryption key that the BGT engine will use when trying to decrypt a sound that has previously been
encrypted. The key must match exactly that which was used when originally encrypting the sound, or it will fail to load.

It is perfectly legal to call this function more than once in your game. Sounds that are already loaded will not be affected, any
changes are only seen the next time you attempt to load a sound.

If the force_encryption parameter is set to true, any sound that BGT finds not to be encrypted will fail to load. This is useful if
you do not wish users to replace your sounds with new files, as these new files would not be encrypted using your key and
would thus not load. If this parameter is set to false, both encrypted and unencrypted sounds will work equally well.

Example:
// Set a decryption key, and tell the engine not to load sounds that have not been encrypted.

void  main()
{
set_sound_decryption_key("please  don't  shoot  me",  true);
}

set_sound_master_volume

This function sets the master volume for the sound mixer.

bool set_sound_master_volume(double volume)

Parameters:
volume
The new value that is to be the master volume, between a maximum of 0 and a minimum of -100.

Return value:
true on success, false on failure.

Remarks:
The master volume works similar to the master fader on a mixing desk. It controls every sound that is played during the game
and works independently of a sound object's volume control. This means that you can continue treating the volume of individual
sound objects exactly the same as before, where 0 is always the maximum based on the current master volume. This can be
useful if your game uses quiet TTS output and has a rather loud or busy soundscape.

Please note that these functions do not apply to the Windows volume control, only sounds that are played from within the game
itself.

Example:
// Set the master volume to -30 and retrieve the new value.

void  main()
{
set_sound_master_volume(-30);
alert("Sound  master  volume",  "The  sound  master  volume  is  "  +  get_sound_master_volume()  +  ".");
}

set_sound_storage

This function sets the location of the pack file in which the engine will look for sound files loaded through the sound object.

bool set_sound_storage(string filename)

Parameters:
filename
The pack file to use. This can be either an absolute path, a relative path, the file name on its own, or a * (see remarks).

Return value:
true on success, false on failure.

Remarks:
Note that both \ and / can be used to specify paths.

If an empty string is passed to this function, the engine will not look for sounds in a pack file but will attempt to locate requested
files on disk instead. This is the default behavior.

If you pass * to this function, the engine will attempt to look for sounds in a pack file that has been included into the program
itself. For more information on how this is done, see the tutorial on packaging files with your executable.

It is perfectly legal to call this function more than once in your game. Sounds that are already loaded will not be affected, any
changes are only seen the next time you attempt to load a sound.

Example:
// Tell the engine to look for sound files in a file called sounds.dat.

void  main()
{
set_sound_storage("sounds.dat");
}

assert

This function evaluates an expression and terminates the script execution if the result is false.

bool assert(bool expression)

Parameters:
expression
The expression to be evaluated.

Return value:
true if the assertion evaluates to true, false otherwise.

Remarks:
This function can be used when you want to be certain that a particular condition is true, and report a problem if it is not. This is
useful mainly for debugging purposes, as the function shows up a message window with the exact line number and source file in
which an assertion failure occurs. You are then able to determine the exact expression that failed, and where in the source code
it's located.

In the message window that appears when an assertion failure occurs, you are able to choose whether you wish to resume
execution or exit the program altogether. If you resume execution, the return value from this function will be false. This allows you
to check in the application itself whether or not a particular assertion call failed.

Please note that this function does nothing if called from a release build of your game. It works only when the script is run from
source, or when a debug build is used. When called from a release build, the function always returns true.

Example:
// Chek a few expressions and finally cause an assertion failure to occur.

void  main()
{
int  x=1;
assert(x<3);
x+=1;
assert(x<3);
x+=1;
assert(x<3);  // We fail here.
}

generate_profile

This function generates the log file containing the results of the profiling (see remarks).

bool generate_profile(string filename)

Parameters:
filename
The name of the file to write to.

Return value:
true on success, false on failure.

Remarks:
The BGT profiler follows the code's execution chain and times each function that is called throughout the game. The profiler will then write a log of all the accumulated
statistics, which include the overall length of time each function took to execute throughout the game's lifetime, and a percentage based on this. This is useful for checking to
see if any functions need further optimisation in order to improve overall performance and speed. For more information about optimising your code, please consult the tutorial
that deals with optimization strategies.

For the profiler to work accurately, you should call start_profiling at the start of your script (usually in main) before any other function is called, and call generate_profile
immediately before your game is due to exit.

This function will generate the log file, stop the profiler, and reset it.

Please note that when you profile debug builds or source code, the results are considerably more accurate as compared with release builds. In debug builds and when running
source code, some execution speed is sacrificed for the benefit of being able to diagnose performance bottlenecks and other problems more easily. This is not the case with
release builds which means that the results that the profiler produces, while not entirely inaccurate, should not be relied upon.

Example:
/*
Write a series of functions that will take different speeds and have the profiler give us some information.
lets_go and lets_finish_here will of course show varying results depending on how long it takes the user to close the message box.
On a system using a 1.8 GHZ duel core processor, the results show as follows:

void  lets_do_some_more():  8055  ms  (76.9%)
void  lets_finish_here():  1122  ms  (10.71%)
void  another_loop():  879  ms  (8.39%)
void  lets_go():  418  ms  (3.99%)

*/

void  main()
{
start_profiling();
lets_go();
lets_do_some_more();
lets_finish_here();
}

void  lets_go()
{
alert("Hi!",  "This  program  demonstrates  the  use  of  the  BGT  profiler.");
}

void  lets_do_some_more()
{
string  test;
for(int  counter=0;  counter<13579000;  counter++)
{
test+="a";
if(counter%1000==0)

{
test="";
}
}
another_loop();
}

void  another_loop()
{
for(int  counter=0;  counter<13500000;  counter++)
{
int  check=1000;
}
}

void  lets_finish_here()
{
alert("Finished.",  "The  profiler  will  now  generate  the  log.");
generate_profile("profiler.log");
}

get_call_stack

This function returns information about the functions that are currently on the stack, as a formatted string.

string get_call_stack()

Parameters:
None.

Return value:
A string containing information about the functions that are on the call stack (see remarks).

Remarks:
The call stack is a list of the functions that are being executed recursively. The function will return the stack size, and various information about each
function, including the script file and line number from which the function will continue executing once control is returned to it.

Please note that only function names can be retrieved if this function is called in a release build of your game.

Example:
/*
This example is deliberately written to be tedious to follow, and thus showing the usefulness of tracing the stack.
*/

class  dummy
{
void  first_dummy_method()
{
second_dummy_method();
}

void  second_dummy_method()
{
clipboard_copy_text(get_call_stack());
}
}

void  main()
{
second();
}

int  second()
{
third();
return  0;
}

void  third()
{
dummy  temp;
temp.first_dummy_method();
}

get_call_stack_size

This function returns the number of functions that are currently on the stack.

double get_call_stack_size()

Parameters:
None.

Return value:
The number of functions that are on the call stack (see remarks).

Remarks:
The call stack is a list of the functions that are being executed recursively. This function is useful when you wish to check how many functions that are currently on the stack, without retrieving all the information about
each one.

Example:
/*
This example makes a few recursive function calls and then prints the stack size.
*/

int  number_of_calls_to_make=100;

void  main()
{
counter_function();
}

void  counter_function()
{
number_of_calls_to_make-=1;
if(number_of_calls_to_make==0)
{
alert("Stack  size",  "The  stack  size  is  "  +  get_call_stack_size()  +  ".");
/*
The output should be 101, as main is also being called. After this point, we return back through the entire chain of functions and finally exit when we get to main.
*/
return;
}
counter_function();
}

get_last_error

This function retrieves the error code set by the last engine function or object method that you called.

int get_last_error()

Parameters:
None.

Return value:
One of the BGT error codes, see appendix B for a full list.

Remarks:
Many engine functions and object methods cannot give any specific information about an error that it encountered via its return value, such as the input_box function
which returns an empty string if an error occurs. This, however, can also happen if the user just clicks OK which makes it impossible to distinguish these two
conditions from one another. This is when the get_last_error function comes in handy. It will return an error code that the engine function or object method specified
before returning, which allows you to determine with much greater exactitude what went wrong.

The get_last_error function will return 0 if no error occured in the last call. This means that the error is reset for every new call to an engine function or object
method, so you should retrieve the error code before making any other calls.

The error code returned by the get_last_error function is not affected by calls to any of your own functions or methods.

Example:
//  Generate  an  input  box,  and  check  whether  the  user  pressed  cancel  by  calling  get_last_error.

void  main()
{
string  name=input_box("Name",  "Please  type  in  your  name");
if(name=="")
{
if(get_last_error()==-12)
{
alert("OK","If  you  wish  to  press  cancel,  then  I  will  trouble  you  no  further.  Goodbye!");
exit();
}
alert("OK","If  you  wish  not  to  disclose  that  personal  precious  information,  then  I  will  trouble  you  no  further.  Goodbye!");
exit();
}
alert("Hello!",  "Hello  "+name+",  nice  to  meet  you!");
}

get_last_error_text

This function retrieves the textual error description set by the last engine function or object method that you called.

string get_last_error_text()

Parameters:
None.

Return value:
An error description if an error occured in the last call, or a blank string otherwise.

Remarks:
Many engine functions and object methods cannot give any specific information about an error that it encountered via its return value, such as the
input_box function which returns an empty string if an error occurs. This, however, can also happen if the user just clicks OK which makes it impossible
to distinguish these two conditions from one another. This is when the get_last_error_text function comes in handy. It will return a textual error description
that the engine function or object method specified before returning, which allows you to determine with much greater exactitude what went wrong.

The get_last_error_text function will return a blank string if no error occured in the last call. This means that the error is reset for every new call to an
engine function or object method, so you should retrieve the error description before making any other calls.

The error description returned by the get_last_error_text function is not affected by calls to any of your own functions or methods.

Example:
// Attempt to open a file that we can be fairly sure does not exist, and print the output from get_last_error_text.

void  main()
{
file  reader;
reader.open("a_file_that_is_dead");
alert("Result",  get_last_error_text());
}

set_error_output

This function logs a list of internal errors set by the engine during the script's execution and writes them to a specified file.

bool set_error_output(string filename)

Parameters:
filename
The name of the file to write to.

Return value:
true on success, false on failure.

Remarks:
If an internal error is encountered while this function is active, it will print the engine function that produced the error, the error description, and the stack
trace of the script into the log file of your choice. This is useful for debugging whether certain unexpected error conditions are being met and are not
accounted for in the script.

Please note that runtime errors are not accounted for, only errors that can be retrieved from within the script with the get_last_error or get_last_error_text
functions.

For this function to work accurately throughout the whole script's execution, it is recommended to call it before calling any other function, in the beginning of
main.

Example:
// Generate a bunch of error conditions and have the program log them.

void  main()
{
timer  error;
sound  test;
file  whatever;
set_error_output("errors.log");
whatever.open("test.temp.txt",  "wb");
whatever.write("This  is  a  test.\r\n");
whatever.close();
whatever.open("test.temp.txt",  "rb");
whatever.write("This  shouldn't  work.\r\n");  //  This  should  generate  an  error  because  the  file  is  opened  for  reading.
whatever.close();
file_delete("test.temp.txt");
test.stream("z:\\whatever.mp3.ogg.wav");  //  This  should  generate  an  error,  because  this  file  probably  doesn't  exist.
error.pause();
error.pause();  //  This  should  generate  an  error,  since  the  timer  is  already  paused.
directory_delete("z:/this/directory/should/not/exist/at/all");  //  This  should  generate  yet  another  nice  error  to  log.
}

start_profiling

This function starts the BGT profiler (see remarks).

bool start_profiling()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
The BGT profiler follows the code's execution chain and times each function that is called throughout the game. The profiler will then write a log of all the accumulated
statistics, which include the overall length of time each function took to execute throughout the game's lifetime, and a percentage based on this. This is useful for checking to
see if any functions need further optimisation in order to improve overall performance and speed. For more information about optimising your code, please consult the tutorial
that deals with optimization strategies.

Please note that when you profile debug builds or source code, the results are considerably more accurate as compared with release builds. In debug builds and when running
source code, some execution speed is sacrificed for the benefit of being able to diagnose performance bottlenecks and other problems more easily. This is not the case with
release builds which means that the results that the profiler produces, while not entirely inaccurate, should not be relied upon.

Example:
/*
Write a series of functions that will take different speeds and have the profiler give us some information.
lets_go and lets_finish_here will of course show varying results depending on how long it takes the user to close the message box.
On a system using a 1.8 GHZ duel core processor, the results show as follows:

void  lets_do_some_more():  8055  ms  (76.9%)
void  lets_finish_here():  1122  ms  (10.71%)
void  another_loop():  879  ms  (8.39%)
void  lets_go():  418  ms  (3.99%)

*/

void  main()
{
start_profiling();
lets_go();
lets_do_some_more();
lets_finish_here();
}

void  lets_go()
{
alert("Hi!",  "This  program  demonstrates  the  use  of  the  BGT  profiler.");
}

void  lets_do_some_more()
{
string  test;
for(int  counter=0;  counter<13579000;  counter++)
{
test+="a";
if(counter%1000==0)
{
test="";
}
}
another_loop();
}

void  another_loop()
{
for(int  counter=0;  counter<13500000;  counter++)
{
int  check=1000;
}
}

void  lets_finish_here()
{
alert("Finished.",  "The  profiler  will  now  generate  the  log.");
generate_profile("profiler.log");
}

stop_profiling

This function stops the BGT profiler and prevents it from gathering statistics until it is restarted with the start_profiling function (see remarks).

bool stop_profiling()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
The BGT profiler follows the code's execution chain and times each function that is called throughout the game. The profiler will then write a log of all the accumulated statistics, which include the overall
length of time each function took to execute throughout the game's lifetime, and a percentage based on this. This is useful for checking to see if any functions need further optimisation in order to improve
overall performance and speed. For more information about optimising your code, please consult the tutorial that deals with optimization strategies.

For the profiler to work accurately, you should call start_profiling at the start of your script (usually in main) before any other function is called, and call generate_profile immediately before your game is
due to exit.

This function resets all previously gathered statistics without writing out a log file with the results first.

Please note that when you profile debug builds or source code, the results are considerably more accurate as compared with release builds. In debug builds and when running source code, some
execution speed is sacrificed for the benefit of being able to diagnose performance bottlenecks and other problems more easily. This is not the case with release builds which means that the results that
the profiler produces, while not entirely inaccurate, should not be relied upon.

Example:
/*
Write a series of functions that will take different speeds and have the profiler give us some information. Stop the profiler on the quickest function.
lets_go and lets_finish_here will of course show varying results depending on how long it takes the user to close the message box.

*/

void  main()
{
start_profiling();
lets_go();
lets_do_some_more();
lets_finish_here();
}

void  lets_go()
{
alert("Hi!",  "This  program  demonstrates  the  use  of  the  BGT  profiler.");
}

void  lets_do_some_more()
{
string  test;
for(int  counter=0;  counter<13579000;  counter++)
{
test+="a";
if(counter%1000==0)
{
test="";
}
}
another_loop();
}

void  another_loop()

{
stop_profiling();
for(int  counter=0;  counter<13500000;  counter++)
{
int  check=1000;
}
}

void  lets_finish_here()
{
alert("Finished.",  "The  profiler  will  now  generate  the  log.");
generate_profile("profiler.log");
}

directory_create

This function creates one or more directories.

bool directory_create(string directories)

Parameters:
directories
A list of one or more directories that you wish to create, separated by either slash or backslash.

Return value:
true on success, false on failure.

Remarks:
This function will create a tree with one or more directories. If one or more of the specified new directories already exist, the function will leave these
untouched and report success.

Please note that the list of directories may not end in a trailing slash or backslash, and may not contain wildcards.

The directory tree may be specified as either an absolute or a relative path.

Example:
// Create a directory called dummy in the application's current directory, with a sub folder called test inside it.

void  main()
{
if(directory_create("dummy\\test")==false)
{
alert("Error",  "The  directory  could  not  be  created.");
}
else
{
alert("Success",  "The  directory  was  created  successfully.");
}
}

directory_delete

This function irreversibly deletes the specified directory, including any files and directories that may be present within it.

bool directory_delete(string directory_name)

Parameters:
directory_name
The name of the directory to delete.

Return value:
true on success, false on failure.

Remarks:
Please note that certain attributes on files or directories can cause the deletion to fail. Also, if a file inside the directory is currently
being used by another program, the operation will fail when it gets to this file. This means that the folder will end up in a half
deleted state with some files and directories gone and others remaining.

IMPORTANT! Be very careful when using this function, as it deletes not only the directory specified but all the files and
directories that may be present inside it as well. Thus, calling it with a directory path such as C:\Windows may have disastrous
results.

This function works with both absolute and relative paths.

Please note that the path may not end in a trailing slash or backslash, and may not contain wildcards.

Example:
//  Delete  the  directory  C:\test

void  main()
{
if(directory_delete("C:\\test"))
{
alert("Information",  "The  directory  was  deleted  successfully.");
}
else
{
alert("Error",  "The  directory  could  not  be  deleted.");
}
}

directory_exists

This function checks whether a given directory exists.

bool directory_exists(string directory_name)

Parameters:
directory_name
The name of the directory to search for.

Return value:
true if the directory exists, false otherwise.

Remarks:
This function works with both absolute and relative paths.

Please note that the path may not end in a trailing slash or backslash, and may not contain wildcards.

Example:
//  Check  if  the  Windows  directory  exists.

void  main()
{
if(directory_exists("C:\\windows"))
{
alert("Information",  "Windows  directory  exists.  I'd  be  slightly  concerned  if  it  didn't.");
}
else
{
alert("Oops",  "You  broke  Windows!  Hang  on  though,  how  on  earth  is  it  running?");
}
}

file_copy

This function copies a given file.

bool file_copy(string source, string destination, bool overwrite)

Parameters:
source
The name of the file to copy.
destination
The name of the file to copy to.
overwrite
A boolean specifying whether the source file should be overwritten (see remarks).

Return value:
true on success, false on failure.

Remarks:
Situations in which this function may return false could include:

The source file does not exist. In this case the function will set the last error flag to -15.
The destination filename references a directory that does not exist. In this case the last error flag will be set to -30.
The overwrite flag is set to false and the destination exists. In this case the last error flag will be set to -35.

A file copied with this function inherits all the attributes of the source file.

This function works with both absolute and relative paths.

Example:
//  Copy  the  file  C:\test.txt.

void  main()
{
if(file_copy("C:\\test.txt",  "c:\\test_copy.txt",  false))
{
alert("Information",  "The  file  was  copied  successfully.");
}
else
{
alert("Error",  "The  file  could  not  be  copied.\nReason:  "  +  get_last_error_text());
}
}

file_delete

This function irreversibly deletes a given file.

bool file_delete(string filename)

Parameters:
filename
The name of the file to delete.

Return value:
true on success, false on failure.

Remarks:
Please note that certain attributes on the file can cause the deletion to fail. Also, if the file is currently being used by another
program, the operation will fail.

This function works with both absolute and relative paths.

Example:
//  Delete  the  file  C:\test\unusable.txt

void  main()
{
if(file_delete("C:\\test\\unusable.txt"))
{
alert("Information",  "The  file  was  deleted  successfully.");
}
else
{
alert("Error",  "The  file  could  not  be  deleted.");
}
}

file_exists

This function checks whether a given file exists.

bool file_exists(string filename)

Parameters:
filename
The name of the file to search for.

Return value:
true if the file exists, false otherwise.

Remarks:
This function works with both absolute and relative paths.

Example:
//  Check  if  notepad.exe  exists  in  the  Windows  directory.

void  main()
{
if(file_exists("C:\\windows\\notepad.exe"))
{
alert("Information",  "Notepad  exists.");
}
else
{
alert("Oops",  "You  broke  notepad!  It's  sort  of  a  nice  program...");
}
}

find_directories

This function returns all directories found matching a specified search pattern.

string[] find_directories(string search)

Parameters:
search
The search pattern that is to be used when looking for directories.

Return value:
Returns an array of directory names that match the specified search pattern, or an empty array on failure.

Remarks:
Wildcards are accepted in the search term. An asterisk (* sign) indicates 0 or more unknown characters while a question mark
(? sign) represents up to 1 unknown character.

Please note that this function will only find directories. To find files use the find_files function.

If an empty string is passed as a parameter, the function will assume the search term "*" which is to say all folders in the current
directory.

This function will accept absolute or relative paths.

The search term is not case sensitive.

Please note that the path may not end in a trailing slash or backslash.

Example:
// Find all directories on drive C.

void  main()
{
string[]  test;
test=find_directories("C:\\*");
for(uint  counter=0;  counter<test.length();  counter++)
{
alert("array",  "array  "+counter+"="+test[counter]);
}
}

find_files

This function returns all files found matching a specified search pattern.

string[] find_files(string search)

Parameters:
search
The search pattern that is to be used when looking for files.

Return value:
Returns an array of file names that match the specified search pattern, or an empty array on failure.

Remarks:
Wildcards are accepted in the search term. An asterisk (* sign) indicates 0 or more unknown characters while a question mark
(? sign) represents up to 1 unknown character.

Please note that this function will only find files. To find directories, use the find_directories function.

When searching using a 3-character extension, any extension starting with those 3 characters will match, even if it contains
additional characters. For instance, "*.ogg" will match "test.ogg", "sound.oggs" or "bang.oggy", that is of course assuming those
files exist.

If an empty string is passed as a parameter, the function will assume the search term "*" which is to say all files in the current
directory.

This function will accept absolute or relative paths.

The search term is not case sensitive.

Please note that the path may not end in a trailing slash or backslash.

Example:
// Find all files in the windows media folder.

void  main()
{
string[]  test;
test=find_files("C:\\windows/media/*.*");
for(uint  counter=0;  counter<test.length();  counter++)
{
alert("array",  "array  "+counter+"="+test[counter]);
}
}

url_decode

This function decodes a URL compatible string back to its standard form.

string url_decode(string url)

Parameters:
url
The encoded URL string to decode.

Return value:
The decoded string on success, or the passed string on failure.

Remarks:
The url_decode function changes special character combinations in URL's that represent spaces and other symbols back to their ASCII equivalents.

Example:
// Decode a URL.

void  main()
{
alert("Decoded  URL",url_decode("http%3a%2f%2fwww.mysite.com%2ffiles%2fmy+program+%28v1.0%29.exe"));  //  Reads  http://www.mysite.com/files/my  program  (v1.0).exe
}

url_encode

This function encodes a string to be URL compatible.

string url_encode(string url)

Parameters:
url
The URL string to encode.

Return value:
The encoded string on success, or the passed string on failure.

Remarks:
The url_encode function converts special characters that cannot be used on their own to specially adapted combinations that represent that character that are legal. Therefore it is important to use this function
when using any of the HTTP functions for them to work properly.

Example:
// Encode a URL.

void  main()
{
alert("encoded  url",url_encode("http://www.mysite.com/files/my  program  (v1.0).exe"));  //  Reads  http%3a%2f%2fwww.mysite.com%2ffiles%2fmy+program+%28v1.0%29.exe
}

url_get

This function retrieves the contents of a URL on the Internet and returns it as a string.

string url_get(string url)

Parameters:
url
The URL of the file on the Internet that should be retrieved.

Return value:
The data from the server on success, or an error description or blank string on failure (see remarks).

Remarks:
This function will wait until all data has been retrieved from the server before returning. Thus, no large amounts of data should be
downloaded using this function as your game will stop responding completely while the operation is in progress. This function is
appropriate for actions such as posting scores, where the returned content is minimal.

This function will send what is known as a GET request. A GET request is used in most cases when a normal URL is being
visited. The URL given may contain so called GET parameters, which are used to pass small amounts of data to the server.

The URL may not contain any special characters such as spaces or similar. If you wish to include such characters, always
encode the URL by calling the url_encode function before attempting to retrieve it from the Internet.

If the http server to which the URL points uses a port other than 80, you will need to include the port after the host name
proceeded by a colon. The path on the server (if any) should then follow, proceeded by a slash. Such a URL might look like
http://myserver.com:8080/myfile.html.

If multiple calls are made to this function with URL's pointing to the same server, the function will attempt to keep the connection
alive for a short while which results in a dramatic speed increase for subsequent calls.

If the network subsystem fails to initialize, the get_last_error function will return -27 and the returned string will be blank.

If an Internet related error occurs, the get_last_error function will return -29 and the returned string will contain a textual
description of the error.

This function can be used to retrieve both binary and textual data.

Example:
//  Retrieve  the  contents  of  the  index  page  at  blastbay.com.

void  main()
{
string  body=url_get("http://www.blastbay.com/");
if(get_last_error()!=0)
{
alert("Error",  "An  error  occured.\nDescription:  "  +  body  +  "");
}
else
{
clipboard_copy_text(body);
alert("Success",  "The  data  was  retrieved  and  has  been  copied  to  your  clipboard.");
}
}

url_post

This function retrieves the contents of a URL on the Internet and returns it as a string.

string url_post(string url, string parameters)

Parameters:
url
The URL of the file on the Internet that should be retrieved.
parameters
The parameters that should be passed to the server.

Return value:
The data from the server on success, or an error description or blank string on failure (see remarks).

Remarks:
This function will wait until all data has been retrieved from the server before returning. Thus, no large amounts of data should be
downloaded using this function as your game will stop responding completely while the operation is in progress. This function is
appropriate for actions such as posting scores, where the returned content is minimal.

This function will send what is known as a POST request. A POST request is used in those cases where the data you are
sending is longer than is feasible with a GET request.

The URL may not contain any special characters such as spaces or similar. If you wish to include such characters, always
encode the URL by calling the url_encode function before attempting to retrieve it from the Internet.

If the http server to which the URL points uses a port other than 80, you will need to include the port after the host name
proceeded by a colon. The path on the server (if any) should then follow, proceeded by a slash. Such a URL might look like
http://myserver.com:8080/myfile.html.

If multiple calls are made to this function with URL's pointing to the same server, the function will attempt to keep the connection
alive for a short while which results in a dramatic speed increase for subsequent calls.

If the network subsystem fails to initialize, the get_last_error function will return -27 and the returned string will be blank.

If an Internet related error occurs, the get_last_error function will return -29 and the returned string will contain a textual
description of the error.

This function can be used to retrieve both binary and textual data.

Example:
// Post some data to a dummy site.

void  main()
{
string  body=url_post("http://www.mysite.com/test.php",  "Data=Test&ID=495&Name=Example");
if(get_last_error()!=0)
{
alert("Error",  "An  error  occured.\nDescription:  "  +  body  +  "");
}
else
{
clipboard_copy_text(body);
alert("Success",  "The  data  was  retrieved  and  has  been  copied  to  your  clipboard.");
}
}

absolute

This function calculates the absolute value of a given number.

double absolute(double x)

Parameters:
x
The number to calculate.

Return value:
The absolute value of x.

Remarks:
None.

Example:
void  main()
{
alert("absolute",  "The  absolute  value  of  3.1416  is  "+absolute(3.1416)+".");
alert("absolute",  "The  absolute  value  of  -10.6  is  "+absolute(-10.6)+".");
}

arc_cosine

This function returns the principal value of the arc cosine of x, expressed in radians.

double arc_cosine(double x)

Parameters:
x
The value to calculate between -1 and 1.

Return value:
Principal arc cosine of x, in the interval [0,pi] radians. If x is outside the valid range, the function will return 0.

Remarks:
In trigonometrics, arc cosine is the inverse operation of cosine.

Example:
void  main()
{
alert("arc_cosine  test",  "Arc  Cosine  of  -1  is  "+arc_cosine(-1)+".");
alert("arc_cosine  test",  "Arc  Cosine  of  0  is  "+arc_cosine(0)+".");
alert("arc_cosine  test",  "Arc  Cosine  of  1  is  "+arc_cosine(1)+".");
}

arc_sine

This function returns the principal value of the arc sine of x, expressed in radians.

double arc_sine(double x)

Parameters:
x
The number to calculate between -1 and 1.

Return value:
Arc sine of x, in the interval [-pi/2,+pi/2] radians. If x is out of the valid range, this function will return 0.

Remarks:
In trigonometrics, arc sine is the inverse operation of sine.

Example:
void  main()
{
alert("arc_sine  test",  "Arc  sine  of  -1  is  "+arc_sine(-1)+".");
alert("arc_sine  test",  "Arc  sine  of  0  is  "+arc_sine(0)+".");
alert("arc_sine  test",  "Arc  sine  of  1  is  "+arc_sine(1)+".");
}

arc_tangent

This function returns the principal value of the arc tangent of x, expressed in radians.

double arc_tangent(double x)

Parameters:
x
The number to calculate.

Return value:
Principal arc tangent of x, in the interval [-pi/2,+pi/2] radians.

Remarks:
In trigonometrics, arc tangent is the inverse operation of tangent.

Notice that because of the sign ambiguity, a function cannot determine with certainty in which quadrant the angle falls only by its
tangent value.

Example:
void  main()
{
alert("arc_tangent  test",  "Arc  tangent  of  -1  is  "+arc_tangent(-1)+".");
alert("arc_tangent  test",  "Arc  tangent  of  0  is  "+arc_tangent(0)+".");
alert("arc_tangent  test",  "Arc  tangent  of  1  is  "+arc_tangent(1)+".");
}

ceiling

This function returns the smallest integral value that is not less than x.

double ceiling(double x)

Parameters:
x
The number to round.

Return value:
The smallest integral value that is not less than x.

Remarks:
This function will always increase x, except where x is already an integer. For example, the ceiling of 5.1 will be 6, but the ceiling
of -5.1 will be -5.

Example:
void  main()
{
alert("ceiling  test",  "ceiling  of  5.1  is  "+ceiling(5.1)+".");
alert("ceiling  test",  "ceiling  of  5  is  "+ceiling(5)+".");
alert("ceiling  test",  "ceiling  of  -5.1  is  "+ceiling(-5.1)+".");
}

cosine

This function returns the cosine of an angle of x radians.

double cosine(double x)

Parameters:
x
a value representing an angle expressed in radians that will be calculated.

Return value:
The cosine of x.

Remarks:
None.

Example:
void  main()
{
double  param=60.0;
double  result=cosine(param*3.14159/180);
alert("cosine  test",  "The  cosine  of  "+param+"  is  "+result+".");
}

exponent

This function returns the base-e exponential function of x, which is the e number raised to the power x.

double exponent(double x)

Parameters:
x
The number to calculate.

Return value:
Exponential value of x.

Remarks:
If the magnitude of the result is so large that it cannot be represented, the function returns 0.

Example:
void  main()
{
double  param=5.0;
double  result=exponent(param);
alert("exponent  test",  "the  exponential  value  of  "+param+"  is  "+result+".");
}

floor

This function returns the largest integral value that is not greater than x.

double floor(double x)

Parameters:
x
The number to round.

Return value:
The largest integral value not greater than x.

Remarks:
This function will always decrease x, except where x is already an integer. For example, the floor of 5.9 will be 5, but the floor of
-5.9 will be -6.

Example:
void  main()
{
alert("floor  test",  "floor  of  5.9  is  "+floor(5.9)+".");
alert("floor  test",  "floor  of  5  is  "+floor(5)+".");
alert("floor  test",  "floor  of  -5.9  is  "+floor(-5.9)+".");
}

log

This function Returns the natural logarithm of x.

double log(double x)

Parameters:
x
The number to convert.

Return value:
Natural logarithm of x, for values of x greater than zero. If x is 0 or less, the function will return 0.

Remarks:
The natural logarithm is the base-e logarithm, the inverse of the natural exponential function (exponent). For base-10 logarithms,
please see log10.

Example:
void  main()
{
double  param=5.5;
double  result=log(param);
alert("log  test",  "log  "+param+"  is  "+result+".");
}

log10

This function returns the common (base-10) logarithm of x.

double log10(double x)

Parameters:
x
The number to convert.

Return value:
Common logarithm of x, for values of x greater than zero. If x is 0 or less, the function will return 0.

Remarks:
None.

Example:
void  main()
{
double  param=1000.0;
double  result=log10(param);
alert("log10  test",  "log10  "+param+"  is  "+result+".");
}

power

This function raises x to the power y.

double power(double x, double y)

Parameters:
x
The base of the calculation.
y
The exponent of the calculation.

Return value:
Returns x to the power y.

Remarks:
If the magnitude of the result is so large that it cannot be represented, the function returns 0.

Example:
void  main()
{
alert("power  test",  "3  to  the  power  4  is  "+power(3,  4)+".");
}

random

This function generates a random number within a specified range.

long random(long min, long max)

Parameters:
min
The smallest number that can be generated.
max
The largest number that can be generated. This must be equal to or greater than min.

Return value:
A random number between min and max on success, or 0 on failure.

Remarks:
If max is less than or equal to min, min will be returned.

The minimum number that can be generated is -999999999, and the maximum is 999999999.

Example:
// Rolling a pair of dice.

void  main()
{
int  die1=random(1,  6);
int  die2=random(1,  6);
int  total=die1  +  die2;
alert("Rolling  dice",  "You  rolled  a  "  +  die1  +  "  and  a  "  +  die2  +  ",  for  a  total  of  "  +  total  +  "!");
}

random_get_state

This function retrieves the current state of the internal random number generator.

string random_get_state()

Parameters:
None.

Return value:
A string containing the current state of the random number generator on success, or an empty string on failure.

Remarks:
A computer can never be truly random. This is why the name of the family of algorithms that deal with randomness is pseudo random
number generators. Pseudo random number generators work by taking an initial random seed from the user, which is supposed to be a
unique data chunk. This is used by the generator to produce a subsequent series of numbers that seem to be random. However, if the
same seed is passed to the generator it'll produce the same series of numbers each time. By saving and loading the current state of the
random number generator, we can recreate the same series of random numbers starting from any given point in time. This is useful if
you have a simulation based on randomness, for instance, and you wish to be able to recreate the exact same series of events.

When BGT starts it automatically initializes the random number generator with a unique seed, so the initial state of the generator will be
different each time the engine is launched. Therefore, if you wish to reproduce the same series of numbers you must remember the state
from which you want to continue and explicitly set the generator to this state.

Example:
void  main()
{

//  Begin  by  retrieving  the  state  from  the  random  number  generator.
string  initial_state=random_get_state();

// Generate three random numbers between 1 and 10.
alert("Numbers", random(1, 10) + ", " + random(1, 10) + ", " + random(1, 10) + ".");

// Restore the random number generator to the old state that we've saved.
random_set_state(initial_state);

// Generate three new random numbers between 1 and 10. Notice that they are the exact same as before.
alert("Numbers", random(1, 10) + ", " + random(1, 10) + ", " + random(1, 10) + ".");
}

random_set_state

This function sets the state of the internal random number generator to the previously saved state contained in the given string.

bool random_set_state(string state)

Parameters:
state
A string containing a state previously retrieved with the random_get_state function, not necessarily during the same execution round.

Return value:
true on success, false on failure.

Remarks:
A computer can never be truly random. This is why the name of the family of algorithms that deal with randomness is pseudo random
number generators. Pseudo random number generators work by taking an initial random seed from the user, which is supposed to be a
unique data chunk. This is used by the generator to produce a subsequent series of numbers that seem to be random. However, if the
same seed is passed to the generator it'll produce the same series of numbers each time. By saving and loading the current state of the
random number generator, we can recreate the same series of random numbers starting from any given point in time. This is useful if
you have a simulation based on randomness, for instance, and you wish to be able to recreate the exact same series of events.

When BGT starts it automatically initializes the random number generator with a unique seed, so the initial state of the generator will be
different each time the engine is launched. Therefore, if you wish to reproduce the same series of numbers you must remember the state
from which you want to continue and explicitly set the generator to this state.

Example:
void  main()
{

//  Begin  by  retrieving  the  state  from  the  random  number  generator.
string  initial_state=random_get_state();

// Generate three random numbers between 1 and 10.
alert("Numbers", random(1, 10) + ", " + random(1, 10) + ", " + random(1, 10) + ".");

// Restore the random number generator to the old state that we've saved.
random_set_state(initial_state);

// Generate three new random numbers between 1 and 10. Notice that they are the exact same as before.
alert("Numbers", random(1, 10) + ", " + random(1, 10) + ", " + random(1, 10) + ".");
}

round

This function rounds a value to its nearest equivalent based on the number of decimal places you specify.

double round(double value, int decimal_places)

Parameters:
value
The value to round.
decimal_places
The number of decimal places to round to.

Return value:
The specified value rounded to the specified number of decimal places.

Remarks:
This function will round the value up or down where appropriate. The decimal_places parameter controls how the function
should round. A value greater than 0 specifies a number of decimal places, 0 specifies the nearest integer, and negative values
will round to the nearest tens, hundreds, thousands etc.

Example:
void  main()
{
alert("round  test",  "5.31  rounded  to  1  decimal  place  is  "+round(5.31,  1)+".");
alert("round  test",  "5.37  rounded  to  1  decimal  place  is  "+round(5.37,  1)+".");
alert("round  test",  "5.73  rounded  to  the  nearest  integer  is  "+round(5.73,  0)+".");
alert("round  test",  "5.31  rounded  to  the  nearest  ten  is  "+round(5.31,  -1)+".");
}

sine

This function returns the sine of an angle of x radians.

double sine(double x)

Parameters:
x
a value representing an angle expressed in radians that will be calculated.

Return value:
The sine of x.

Remarks:
None.

Example:
void  main()
{
double  param=30.0;
double  result=sine(param*3.14159/180);
alert("sine  test",  "The  sine  of  "+param+"  is  "+result+".");
}

square_root

This function calculates the square root of a number.

double square_root(double value)

Parameters:
value
The value to calculate.

Return value:
The square root of a number.

Remarks:
The square root is the inverse operation of value squared (value*value).

Example:
void  main()
{
alert("square_root  test",  "the  square  root  of  144  is  "+square_root(144)+".");
}

tangent

This function returns the tangent of an angle of x radians.

double tangent(double x)

Parameters:
x
a value representing an angle expressed in radians that will be calculated.

Return value:
The tangent of x.

Remarks:
None.

Example:
void  main()
{
double  param=45.0;
double  result=tangent(param*3.14159/180);
alert("tangent  test",  "The  tangent  of  "+param+"  is  "+result+".");
}

clipboard_copy_text

This function copies text to the clipboard.

bool clipboard_copy_text(string text)

Parameters:
text
The text that should be copied.

Return value:
true on success, false on failure.

Remarks:
This function will copy the specified text to the Windows clipboard, overwriting whatever is already there.

Example:
void  main()
{
clipboard_copy_text("Hello,  I  have  been  automatically  copied!");
}

clipboard_read_text

This function reads text from the clipboard.

string clipboard_read_text()

Parameters:
None.

Return value:
The text stored in the clipboard on success, or an empty string on failure.

Remarks:
None.

Example:
void  main()
{
clipboard_copy_text("Hello,  I  have  been  automatically  copied!");
alert("Clipboard  text",  clipboard_read_text());
}

exit

This function exits the program.

void exit()

Parameters:
None.

Return value:
None.

Remarks:
It is safe to call this function from anywhere within your script. All memory and resources that have been allocated for your game
will be automatically released back to Windows. This also happens whenever a script finishes executing, so it is not necessary to
call this function at the end of your code.

Example:
// Display a message box and then exit.

void  main()
{
alert("Exiting  program",  "Thanks  for  playing!  Ciao  baby!");
exit();
}

garbage_collect

This function runs a complete cycle of the garbage collector used internally in the BGT engine (see remarks).

void garbage_collect()

Parameters:
None.

Return value:
None.

Remarks:
Garbage collection is used in BGT in order to resolve circular references. Circular references are objects that point back to one another and thus make the technique used for the most part in
the engine, reference counting, impractical. Any object that has the potential of creating a circular reference is automatically made known to the garbage collector, such as arrays and script
defined classes. The garbage collector runs incrementally in the background all the time during the execution of your game, and cleans up garbage as soon as it is found. This means that you
should not depend on a destructor of any given class being called at any specific time.

If your script accumulates a lot of garbage and you see memory usage beginning to increase too much, it is a good idea to call this function in order to force the garbage collector to run a
complete cycle and destroy all currently known garbage. This may take up to a few hundred milliseconds and so should only be done at a point in the game when speed is not critical, such as
during a game over sequence or cut-scene.

If BGT notices that enough garbage has been accumulated to cause potential problems such as crashes or program instability, it automatically increases the rate at which the incremental garbage
collection cycle is run. This is not the same as running a complete cycle, but it may impact performance slightly. Thus, calling this function yourself at a time when performance won't suffer is
useful in more ways than one in this scenario as it not only decreases memory usage significantly, but also gives you control over when this time cost should be incurred.

Example:
//  This  script  shows  how  the  garbage  collector  destroys  objects  with  circular  references  incrementally.
// A lot of temporary class instances are created, and then a loop runs to wait for these to be destroyed.
// Once the last instance has been destroyed, the script shows an alert box telling you how long this took and then exits.
// The user can press space to force a complete garbage collection cycle, or escape to exit.

const  int  number_of_instances=50;
// This indicates how much garbage that should be generated. You may change this to experiment.

int  destroyed_instances=0;

class  dummy
{

timer  counter;
// This timer keeps track of how long the class has been in existence.

dummy@  other_dummy;
// This is a handle that is set by the constructor to point to itself. This creates a circular reference.

dummy()
{
counter.restart();
@other_dummy=this;
}

~dummy()
{
destroyed_instances+=1;
if(destroyed_instances==number_of_instances)
{
alert("Destroying  last  class",  "The  last  class  instance  is  now  being  destroyed  after  having  existed  for  "  +  counter.elapsed  +  "  milliseconds.");
exit();
}
}

}

void  main()
{
show_game_window("Garbage  collection  test");

alert("Welcome",  "We  are  now  generating  a  bunch  of  garbage  class  instances  with  circular  references  in  them.");

// Call a function that generates a bunch of temporary instances of the dummy class.
generate_garbage();

while(key_pressed(KEY_ESCAPE)==false)
{
if(key_pressed(KEY_SPACE))
{
garbage_collect();
}
wait(5);
}
}

void  generate_garbage()
{
dummy[]  dummy_list(number_of_instances);
}

hide_game_window

This function hides the main window of your game on the user's screen.

bool hide_game_window()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
When the game window is invisible, the user is not able to set focus to it. This means that you will not be able to accept keyboard input before you call the
show_game_window function.

Example:
// Display the window for 3 seconds, hide it for 3 seconds, display it again for 3 seconds with another title and then exit.

void  main()
{
show_game_window("An  Appalling  Alien  Attack");
wait(3000);
hide_game_window();
wait(3000);
show_game_window("Another  Appalling  Alien  Attack");
wait(3000);
}

is_admin

This function checks whether the game is being run with administrator privileges.

bool is_admin()

Parameters:
None.

Return value:
true if the game is being run with administrator privileges, false otherwise.

Remarks:
Many users of Windows XP and earlier tended to run as administrators all the time. This opened up a lot of security issues that
virus makers could exploit. Therefore, with Windows Vista and above Microsoft made the user rights management a lot more
prominent in the operating system. One example of this is the inclusion of UAC (user access control), which advises users when
programs attempt to do things that could be considered harmful such as writing to certain directories on the file system. With this
function you may check if the current user is running as an administrator, or alternatively if they have enabled administrative
privileges for your process specifically.

Example:
void  main()
{
if(is_admin())
{
alert("Congratulations",  "You  are  an  administrator!");
}
else
{
alert("Sorry",  "You  don't  appear  to  be  privileged  enough  to  be  an  administrator  on  this  system.");
}
}

is_game_window_active

This function checks whether your game window is currently in focus.

bool is_game_window_active()

Parameters:
None.

Return value:
true if your game window is currently active, false otherwise.

Remarks:
This function returns true if your game window has keyboard focus at the present time. It is imperative that you call
show_game_window before you rely on this function.

Example:
// Play a looping sound only while the game window has focus.

void  main()
{
show_game_window("Temperamental  Sound");
sound  noise;
noise.load("myfile.ogg");
if(noise.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
while(true)
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
if(is_game_window_active()==true)
{
if(noise.playing==false)
{
noise.play_looped();
}
}
else
{
if(noise.playing==true)
{
noise.pause();
}
}
wait(5);
}
}

read_environment_variable

This function reads the value of a user specific or system wide environment variable.

string read_environment_variable(string name)

Parameters:
name
The name of the environment variable to read.

Return value:
A string containing the retrieved value on success, or an empty string on failure.

Remarks:
Environment variables are pieces of information that are either related to the operating system or to the current user. This function
can retrieve both types.

The environment variable names are not case sensitive.

Example:
//  Try  to  retrieve  some  information  about  the  processor.

void  main()
{
string  value=read_environment_variable("PROCESSOR_IDENTIFIER");
if(get_last_error()!=0)
{
alert("Error",  get_last_error_text());
exit();
}
alert("Value",  value);
}

run

This function runs an executable program.

bool run(string filename, string command_line, bool wait_for_completion, bool background)

Parameters:
filename
The name of the executable file to run.
command_line
The command line parameters that should be given to the program.
wait_for_completion
A boolean (true or false) that specifies whether or not the function should wait for the new executable to finish running before
returning.
background
A boolean (true or false) that specifies whether or not the new program should be run in the background, e.g. display no
window.

Return value:
true on success, false on failure.

Remarks:
No command line parameters should be specified as part of the executable name. Instead, those should be given as the second
parameter only.

This function works with both absolute and relative paths.

Example:
// Run notepad.exe and wait for it to close.

void  main()
{
bool  result=run("C:\\windows\\notepad.exe",  "",  true,  false);
if(result==false)
{
alert("Error",  "notepad.exe  could  not  be  run.");
exit();
}
alert("Goodbye",  "Thank  you  for  using  notepad.  Your  satisfaction  is  not  guaranteed.");
}

show_game_window

This function displays the main window of your game on the user's screen.

bool show_game_window(string title)

Parameters:
title
The title of the window.

Return value:
true on success, false on failure.

Remarks:
Until this function is called your game window will be invisible, thus not allowing the user to set focus to it. This in turn means that
you will not be able to accept keyboard input before you call this function, and so you will usually want to do this at the very
beginning of your script.

It is perfectly legal to call this function more than once during the course of your game's execution, should you wish to change the
title of the window at any given time.

Example:
// Display the window and wait for the user to close it.

void  main()
{
show_game_window("Another  Appalling  Alien  Attack");
while(true)
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
wait(5);
}
}

wait

This function pauses the entire script for a certain number of millisecunds.

bool wait(int milliseconds)

Parameters:
milliseconds
The number of milliseconds to wait. This must not be less than 0.

Return value:
true on success, false on failure.

Remarks:
This function completely interrupts the script for the specified amount of time. This includes all real-time activities such as
checking timers, keyboard input etc. It is thus not advisable to wait for a long time in situations where the program needs to be
responsive. However, waiting for about 5 milliseconds in your main game loop as well as other loops that run for a long time is
highly recommended, as it reduces the CPU usage of your program enormously without affecting its speed. If you fail to do this,
your game will consume 100 % of the CPU at all times.

Example:
// Display a message, wait for three seconds, and then display another one.

void  main()
{
alert("Waiting","About  to  wait  for  three  seconds.");
wait(3000);
alert("Thanks","Thanks  for  waiting  for  me.  Goodbye.");
}

file_decrypt

This function decrypts a file previously encrypted by file_encrypt.

bool file_decrypt(string input_file, string output_file, string encryption_key)

Parameters:
input_file
The name of the file to decrypt.
output_file
The name of the file that will contain the decrypted data.
encryption_key
The key that will be used to decrypt the data. This must be the same key given when the file was encrypted.

Return value:
true if the file was successfully decrypted, false otherwise.

Remarks:
None.

Example:
// Decrypt a sound file and play it.

sound  test;

void  main()
{
if(!file_exists("test.dat"))
{
alert("Error",  "Cannot  find  test.dat.");
return;
}
if(!file_decrypt("test.dat",  "test.wav",  "you_are_not_hacking_into_me"))
{
alert("Error",  "Could  not  decrypt  test.dat.");
return;
}
test.load("test.wav");
if(test.active==false)
{
alert("Error",  "Cannot  play  decrypted  sound.  errorcode="+error);
return;
}
test.play_wait();
}

file_encrypt

This function encrypts a file based on a provided key.

bool file_encrypt(string input_file, string output_file, string encryption_key)

Parameters:
input_file
The name of the file to encrypt.
output_file
The name of the file that will contain the encrypted data.
encryption_key
The key that will be used to encrypt and later decrypt the data.

Return value:
true if the file was successfully encrypted, false otherwise.

Remarks:
This uses the AES-Rijndel 256-bit encryption algorithm, which is one of the most secure algorithms available to date. It has
never been successfully cracked, and is used to protect files stored by various governments.

Files produced by file_encrypt are not compatible with strings produced by string_encrypt.

This function will encrypt any file, from text files to sounds, and the result can be used with the sound object (see the
set_sound_decryption_key function for details).

The encrypted files are a few bytes longer than their decrypted equivalents.

The encrypted data is in binary form.

Example:
// Encrypt a sound file and play it.

sound  test;

void  main()
{
if(!file_exists("test.wav"))
{
alert("Error",  "Cannot  find  test.wav.");
return;
}
if(!file_encrypt("test.wav",  "test.dat",  "you_are_not_hacking_into_me"))
{
alert("Error",  "Could  not  encrypt  test.wav.");
return;
}
set_sound_decryption_key("you_are_not_hacking_into_me",  true);
test.load("test.dat");
if(test.active==false)
{
alert("Error",  "Cannot  play  encrypted  sound.  errorcode="+error);
return;
}
test.play_wait();
}

file_hash

This function generates what is known as a hash for a file on disk, using one of several available algorithms.

string file_hash(string filename, int algorithm, bool binary)

Parameters:
filename
The name of the file from which the hash should be generated.
algorithm
A number indicating which hashing algorithm to use (see remarks).
binary
A boolean specifying whether or not the resulting string should be in binary or hex form, true meaning binary.

Return value:
The generated hash on success, or a blank string on failure.

Remarks:
A hash is a fixed size representation of an arbitrary amount of data. What this means in practice is that you can pass a file of any
size to this function and you will get a string back which is a sort of identifier of the data in the file, always with the same size. It
doesn't matter if the file is 1 kb or 1 gb, the size of the hash will always be the same. The hash will change completely if even one
byte of the file is different, making it very useful for tasks such as integrity verification etc.

The available algorithms are:

1 - SHA256
2 - Sha512

Please note that due to the nature of hashes, some completely different blocks of data will ultimately generate the same hash. This
is an inevitable consequence of trying to convert an arbitrary amount of data to a fixed size representation, since a hash is entirely
different from data compression. However, the frequency at which duplication occurs depends on what hashing algorithm you
use. The 256 bit algorithms are more likely to generate duplicates than their 512 bit counterparts, which are twice as large and
thus have more room for variation.

Example:
// Generate a SHA256 bit hash for a file, as a hex string.

void  main()
{

string  hash=file_hash("some_file.dat",  1,  false);
if(hash=="")
{
alert("Error",  "The  hash  could  not  be  generated.\nError  description:  "  +  get_last_error_text);
exit();
}
alert("Hash",  hash);
}

generate_computer_id

This function generates an ID unique to the current computer that the game is running on.

string generate_computer_id(string application_id, bool hardware_only)

Parameters:
application_id
A string that uniquely identifies your game.
hardware_only
A boolean specifying whether or not the function should only use information linked to the physical hardware in the machine.

Return value:
The generated computer ID on success, or a blank string on failure.

Remarks:
The ID that this function generates is based on several different hardware components of the machine that the game is running on.
It is used often in registration systems where a piece of software is tied to a specific computer. For more information about how
this ID might be used for this purpose, please see the article called Registration in BGT.

Please note that we cannot guarantee that this system will never be cracked. For more information on this, please refer to our
end user license agreement as well as to the conclusion section of the article mentioned above.

The application ID that you specify is used to make the generated ID different from those generated by other BGT games.
Without the application ID, the given ID would be the same for all BGT applications on a given computer. You must make
certain that the application ID is not something that may be guessed by a third party, and that you only change it when you wish
all your generated ID's to be different such as when you release a new major version of your game. Keeping the application ID
as a string constant in your script is not desirable, instead it should be generated on the fly using some scrambling techniques of
your choosing.

The hardware_only flag is used to indicate whether the function should limit itself to using hardware related information
exclusively. If this is set to false then the function also uses information that changes for each installation of Windows, such as the
hard drive serial assigned by the operating system when the drive was last formatted. This is not hard for the end user to change
manually, but provides a much greater likelyhood that the final ID will indeed be unique. If this flag is set to true, then two
computers with very similar or identical hardware may generate the exact same ID for a given application.

Example:
// Generate a unique computer ID for a dummy application.

void  main()
{
alert("Product  ID",  generate_computer_id("my_dummy_app",  false));
}

string_decrypt

This function decrypts a string previously encrypted by string_encrypt.

string string_decrypt(string the_string, string encryption_key)

Parameters:
the_string
The string to decrypt.
encryption_key
The key that will be used to decrypt the data. This must be the same key given when the string was encrypted.

Return value:
The decrypted string on success, or a blank string on failure.

Remarks:
The input string must match the string returned by string_encrypt. If you have converted the string to hexadecimal for printing, it is essential that you
use hex_to_string to convert it back before using string_decrypt.

Example:
// Decrypt a string and print it.

test="881C830C5245F740C578E6C4F4A7DE10F960D10F3E47535CF56F94DF4F2E4CB38E7C35CA0C30C71AED7A1F0F33BB35F2";

void  main()
{
string 
test=hex_to_string(test);
test=string_decrypt(test,  "you_are_not_hacking_into_me");
alert("Decrypted  string",  test);
}

string_encrypt

This function encrypts a string based on a provided key.

string string_encrypt(string the_string, string encryption_key)

Parameters:
the_string
The string to encrypt.
encryption_key
The key that will be used to encrypt and later decrypt the data.

Return value:
The encrypted string on success, or a blank string on failure.

Remarks:
This uses the AES-Rijndel 256-bit encryption algorithm, which is one of the most secure algorithms available to date. It has
never been successfully cracked, and is used to protect files stored by various governments.

Strings produced by string_encrypt are not compatible with files produced by file_encrypt.

This function will encrypt both binary and text-only strings.

The encrypted string is a few bytes longer than its decrypted equivalent.

The encrypted data is in binary form. Thus, if you print out an encrypted string directly, the output will be garbled. If you need to
print out an encrypted string, use the string_to_hex function to convert it into hexadecimal.

Example:
// Encrypt a string and print it.

void  main()
{
string  test=string_encrypt("I  am  an  encrypted  string.",  "you_are_not_hacking_into_me");
test=string_to_hex(test);
alert("Encrypted  string",  test);
}

string_hash

This function generates what is known as a hash for a block of data, using one of several available algorithms.

string string_hash(string data, int algorithm, bool binary)

Parameters:
data
The data from which the hash should be generated.
algorithm
A number indicating which hashing algorithm to use (see remarks).
binary
A boolean specifying whether or not the resulting string should be in binary or hex form, true meaning binary.

Return value:
The generated hash on success, or a blank string on failure.

Remarks:
A hash is a fixed size representation of an arbitrary amount of data. What this means in practice is that you can pass a block of
data of any length to this function and you will get a string back which is a sort of identifier of the data, always with the same size.
It doesn't matter if the data block is 1 kb or 1 gb, the size of the hash will always be the same. The hash will change completely if
even one byte of the data block is different, making it very useful for tasks such as integrity verification of files etc.

The available algorithms are:

1 - SHA256
2 - Sha512

Please note that due to the nature of hashes, some completely different blocks of data will ultimately generate the same hash. This
is an inevitable consequence of trying to convert an arbitrary amount of data to a fixed size representation, since a hash is entirely
different from data compression. However, the frequency at which duplication occurs depends on what hashing algorithm you
use. The 256 bit algorithms are more likely to generate duplicates than their 512 bit counterparts, which are twice as large and
thus have more room for variation.

Example:
// Generate a SHA256 bit hash for a normal piece of text, as a hex string.

void  main()
{
string  text="Hello,  this  is  the  text  from  which  the  hash  should  be  generated.";
alert("Original  text",  text);
text=string_hash(text,  1,  false);
alert("Hash",  text);
}

deserialize

This function deserializes a string previously constructed by serialize.

dictionary@ deserialize(string the_data)

Parameters:
the_data
The data as a string to restore.

Return value:
A handle to a dictionary object with the restored data on success, or an empty dictionary on failure.

Remarks:
Serialization is the process of storing data in the game and converting them to a stream of bytes, which is easily saved to a file and retrieved again later. This can be anything from entire levels with enemies and traps, to
simple configuration data.
BGT's serialization functionality takes away most of the tedium involved in designing a format for saving your data, writing a parser to convert the data back into something understood by the game etc.
Each piece of data has a name which is used when it is time to look up the data again when it needs to be loaded. This means that the order in which things are saved is no longer relevant because each value is looked
up based on its name, rather than based on where in the long list of values it is expected to be stored. Furthermore, when you release a new version of your game which requires new content to be saved, your old data
files will still be compatible with the new version provided that you are able to set the new variables to default values if they are not found.
See the serialisation tutorial for more information.

Example:
//  Deserialize  some  data.

// Create a map class.

class  map
{
int[][]  location;  //  Stores  all  the  various  positions  on  the  map.
int size_x; // The number of squares to place on the X axis.
int size_y; // The number of squares to place on the Y axis.

//  Create  our  constructor.
map()
{
reset();
}

// Implement a reset method for resetting our map.
void  reset()
{
location.resize(0);
size_x=0;
size_y=0;
}

//  Implement  a  rudimentary  setup  procedure.
void  setup(int  max_x=20,  int  max_y=20)
{
if((max_x<=0)||(max_y<=0))
{
return;
}

// Fill out our properties.
size_x=max_x;
size_y=max_y;

// Fill up the array.
location.resize(max_x);
for(int  x=0;  x<max_x;  x++)

{
location[x].resize(max_y);

// Assign values to it.
for(int  y=0;  y<max_y;  y++)
{
location[x][y]=random(0,  5);
}
}
}

// This serializes the map.
string  save()
{
dictionary  data;  //  We  will  pass  this  dictionary  to  the  serialize  function.

// First, we store the size of the map so we know how many squares we have available.
data.set("map.size_x",  size_x);
data.set("map.size_y",  size_y);

// Put our map in the dictionary.
for(int  x=0;  x<size_x;  x++)
{
for(int  y=0;  y<size_y;  y++)
{
data.set("map.location["+x+"]["+y+"]",  location[x][y]);
}
}

//  Return  the  serialization.
return  serialize(data);
}

// This will deserialize and load the values back into the map.
bool  load(string  data)
{

// Check if we have no data.
if(data=="")
{
return  false;
}
dictionary@  restore=deserialize(data);  //  This  is  where  the  data  will  be  stored  initially.
reset(); // So we have a clean start.
restore.get("map.size_x",  size_x);
restore.get("map.size_y",  size_y);
if((size_x<=0)||(size_y<=0))
{
return  false;
}

//Resize  our  map.
location.resize(size_x);
for(int  x=0;  x<size_x;  x++)
{
location[x].resize(size_y);

// Now we can restore our values.
for(int  y=0;  y<size_y;  y++)
{
restore.get("map.location["+x+"]["+y+"]",  location[x][y]);
}
}
return  true;
}
}

// Create a player class

class  player

{
int x;
int y;
int  score;
int  health;

player()
{
reset();
}

void  reset()
{
x=0;
y=0;
health=100;
score=0;
}

void  setup()
{

//  This  will  set  random  values  for  each  property,  merely  to  demonstrate  serialization.
x=random(0,  10);
y=random(0,  10);
score=random(0,  500000);
health=random(1,  100);
}

string  save()
{
dictionary  data;  //  We  will  pass  this  dictionary  to  the  serialize  function.

// Put our values in the dictionary.
data.set("player.x",  x);
data.set("player.y",  y);
data.set("player.health",  health);
data.set("player.score",  score);

//  Return  the  serialization.
return  serialize(data);
}

// This will deserialize and load the values back into the class.
bool  load(string  data)
{

// Check if we have no data.
if(data=="")
{
return  false;
}
dictionary@  restore=deserialize(data);  //  This  is  where  the  data  will  be  stored  initially.
reset(); // So we have a clean start.
restore.get("player.x",  x);
restore.get("player.y",  y);
restore.get("player.health",  health);
restore.get("player.score",  score);
return  true;
}
}

// Create an enemy class

class  enemy
{
int id; //Used to store a unique identifier, typically the position in the array where he is stored.
int x;
int y;
int  speed;

int  health;
int  state;

enemy()
{
reset();
}

void  reset()
{
id=-1;
x=0;
y=0;
speed=random(300,  500);
health=100;
state=0; //Can be used to store whether he is attacking, walking etc.
}

void  setup(int  id)
{

// This will set random values for each property, merely to demonstrate serialization. We ask for the ID so we can serialize the correct enemy when the time comes.
this.id=id;
x=random(0,  10);
y=random(0,  10);
speed=random(300,  500);
health=random(1,  100);
state=random(0,  5);  //Can  be  used  to  store  whether  he  is  attacking,  exploring,  fleeing  etc.
}

string  save()
{
dictionary  data;  //  We  will  pass  this  dictionary  to  the  serialize  function.

//  Put  our  values  in  the  dictionary.  Notice  we  use  the  enemy  ID  as  a  key  rather  than  as  a  value.
data.set("enemy["+id+"].x",  x);
data.set("enemy["+id+"].y",  y);
data.set("enemy["+id+"].health",  health);
data.set("enemy["+id+"].state",  state);
data.set("enemy["+id+"].speed",  speed);

//  Return  the  serialization.
return  serialize(data);
}

// This will deserialize and load the values back into the class.
bool  load(string  data,  int  id)
{

// Check if we have no data.
if((data=="")||(id<0))
{
return  false;
}
dictionary@  restore=deserialize(data);  //  This  is  where  the  data  will  be  stored  initially.
reset(); // So we have a clean start.
this.id=id;
restore.get("enemy["+id+"].x",  x);
restore.get("enemy["+id+"].y",  y);
restore.get("enemy["+id+"].health",  health);
restore.get("enemy["+id+"].state",  state);
restore.get("enemy["+id+"].speed",  speed);
return  true;
}
}

// Now we declare some values and deserialize them from a file.

player  me;
map  world;

enemy[]  monster;

void  main()
{

//  Declare  our  variables.
file  storage;

int  monsters=1;

//Retrieve  our  data.
storage.open("serialize_test.txt",  "rb");
string  data=storage.read();
storage.close();

//  Construct  our  dictionary.
dictionary@  example=deserialize(data);

//Restore  our  data  back  into  our  variables.
example.get("enemy.length",  monsters);
world.load(data);
me.load(data);
monster.resize(monsters);
for(int  counter=0;  counter<monsters;  counter++)
{
monster[counter].load(data,  counter);
}
}

serialize

This function serializes a dictionary into a string.

string serialize(dictionary the_data)

Parameters:
the_data
The data stored in a dictionary.

Return value:
A string with the serialized data on success, or a blank string on failure.

Remarks:
Serialization is the process of storing data in the game and converting them to a stream of bytes, which is easily saved to a file and retrieved again later. This can be anything from entire levels with enemies and traps, to
simple configuration data.
BGT's serialization functionality takes away most of the tedium involved in designing a format for saving your data, writing a parser to convert the data back into something understood by the game etc.
Each piece of data has a name which is used when it is time to look up the data again when it needs to be loaded. This means that the order in which things are saved is no longer relevant because each value is looked
up based on its name, rather than based on where in the long list of values it is expected to be stored. Furthermore, when you release a new version of your game which requires new content to be saved, your old data
files will still be compatible with the new version provided that you are able to set the new variables to default values if they are not found.
See the serialisation tutorial for more information.

Example:
//  Serialize  some  data.

// Create a map class.

class  map
{
int[][]  location;  //  Stores  all  the  various  positions  on  the  map.
int size_x; // The number of squares to place on the X axis.
int size_y; // The number of squares to place on the Y axis.

//  Create  our  constructor.
map()
{
reset();
}

// Implement a reset method for resetting our map.
void  reset()
{
location.resize(0);
size_x=0;
size_y=0;
}

//  Implement  a  rudimentary  setup  procedure.
void  setup(int  max_x=20,  int  max_y=20)
{
if((max_x<=0)||(max_y<=0))
{
return;
}

// Fill out our properties.
size_x=max_x;
size_y=max_y;

// Fill up the array.
location.resize(max_x);
for(int  x=0;  x<max_x;  x++)

{
location[x].resize(max_y);

// Assign values to it.
for(int  y=0;  y<max_y;  y++)
{
location[x][y]=random(0,  5);
}
}
}

// This serializes the map.
string  save()
{
dictionary  data;  //  We  will  pass  this  dictionary  to  the  serialize  function.

// First, we store the size of the map so we know how many squares we have available.
data.set("map.size_x",  size_x);
data.set("map.size_y",  size_y);

// Put our map in the dictionary.
for(int  x=0;  x<size_x;  x++)
{
for(int  y=0;  y<size_y;  y++)
{
data.set("map.location["+x+"]["+y+"]",  location[x][y]);
}
}

//  Return  the  serialization.
return  serialize(data);
}

// This will deserialize and load the values back into the map.
bool  load(string  data)
{

// Check if we have no data.
if(data=="")
{
return  false;
}
dictionary@  restore=deserialize(data);  //  This  is  where  the  data  will  be  stored  initially.
reset(); // So we have a clean start.
restore.get("map.size_x",  size_x);
restore.get("map.size_y",  size_y);
if((size_x<=0)||(size_y<=0))
{
return  false;
}

//Resize  our  map.
location.resize(size_x);
for(int  x=0;  x<size_x;  x++)
{
location[x].resize(size_y);

// Now we can restore our values.
for(int  y=0;  y<size_y;  y++)
{
restore.get("map.location["+x+"]["+y+"]",  location[x][y]);
}
}
return  true;
}
}

// Create a player class

class  player

{
int x;
int y;
int  score;
int  health;

player()
{
reset();
}

void  reset()
{
x=0;
y=0;
health=100;
score=0;
}

void  setup()
{

//  This  will  set  random  values  for  each  property,  merely  to  demonstrate  serialization.
x=random(0,  10);
y=random(0,  10);
score=random(0,  500000);
health=random(1,  100);
}

string  save()
{
dictionary  data;  //  We  will  pass  this  dictionary  to  the  serialize  function.

// Put our values in the dictionary.
data.set("player.x",  x);
data.set("player.y",  y);
data.set("player.health",  health);
data.set("player.score",  score);

//  Return  the  serialization.
return  serialize(data);
}

// This will deserialize and load the values back into the class.
bool  load(string  data)
{

// Check if we have no data.
if(data=="")
{
return  false;
}
dictionary@  restore=deserialize(data);  //  This  is  where  the  data  will  be  stored  initially.
reset(); // So we have a clean start.
restore.get("player.x",  x);
restore.get("player.y",  y);
restore.get("player.health",  health);
restore.get("player.score",  score);
return  true;
}
}

// Create an enemy class

class  enemy
{
int id; //Used to store a unique identifier, typically the position in the array where he is stored.
int x;
int y;
int  speed;

int  health;
int  state;

enemy()
{
reset();
}

void  reset()
{
id=-1;
x=0;
y=0;
speed=random(300,  500);
health=100;
state=0; //Can be used to store whether he is attacking, walking etc.
}

void  setup(int  id)
{

// This will set random values for each property, merely to demonstrate serialization. We ask for the ID so we can serialize the correct enemy when the time comes.
this.id=id;
x=random(0,  10);
y=random(0,  10);
speed=random(300,  500);
health=random(1,  100);
state=random(0,  5);  //Can  be  used  to  store  whether  he  is  attacking,  exploring,  fleeing  etc.
}

string  save()
{
dictionary  data;  //  We  will  pass  this  dictionary  to  the  serialize  function.

//  Put  our  values  in  the  dictionary.  Notice  we  use  the  enemy  ID  as  a  key  rather  than  as  a  value.
data.set("enemy["+id+"].x",  x);
data.set("enemy["+id+"].y",  y);
data.set("enemy["+id+"].health",  health);
data.set("enemy["+id+"].state",  state);
data.set("enemy["+id+"].speed",  speed);

//  Return  the  serialization.
return  serialize(data);
}

// This will deserialize and load the values back into the class.
bool  load(string  data,  int  id)
{

// Check if we have no data.
if((data=="")||(id<0))
{
return  false;
}
dictionary@  restore=deserialize(data);  //  This  is  where  the  data  will  be  stored  initially.
reset(); // So we have a clean start.
this.id=id;
restore.get("enemy["+id+"].x",  x);
restore.get("enemy["+id+"].y",  y);
restore.get("enemy["+id+"].health",  health);
restore.get("enemy["+id+"].state",  state);
restore.get("enemy["+id+"].speed",  speed);
return  true;
}
}

// Now we declare some values and serialize them to a file.

player  me;
map  world;

enemy[]  monster;

void  main()
{

//  Declare  our  variables.
dictionary  example;
file  storage;

// Set up our environment.
int  monsters=random(5,  10);
monster.resize(monsters);
for(int  counter=0;  counter<monster.length();  counter++)
{
monster[counter].setup(counter);
}
me.setup();
world.setup();

//  Construct  our  serialization.
example.set("enemy.length",  monsters);
string  data=world.save();
data+=me.save();
data+=serialize(example);
for(int  counter=0;  counter<monsters;  counter++)
{
data+=monster[counter].save();

// We may as well reset our enemies in this loop, since we are going to reset everything after anyway.
monster[counter].reset();
}

//  Go  about  resetting  everything.
monster.resize(0);
me.reset();
world.reset();

// Save our serialization to a file.
storage.open("serialize_test.txt",  "wb");
storage.write(data);
storage.close();
}

alert

This function displays a message box on the user's screen with a title and text.

bool alert(string title, string text)

Parameters:
title
The title of the window.
text
The text that is to be shown.

Return value:
true on success, false on failure.

Remarks:
This function is useful for displaying information to the user such as error messages and other important alerts. The script
execution will pause until the user clicks on the OK button.

Example:
void  main()
{
alert("Information", "Did you know that I am an alert box? No? Well, now you do!");
}

force_key_down

This function causes a given key to assume the state of being held down.

bool force_key_down(int key)

Parameters:
key
One of the BGT key name constants, see appendix A for a full list.

Return value:
true on success, false on failure.

Remarks:
Forcing keys is useful for controlling certain aspects of games. For example, if you wanted a game to completely replay itself in the player's style, you might record the code and length of each key pressed in a previous game, and code the replay so that all the keys are forced up or down at the right moments, thereby completely
simulating the player's actions.

The state of the key being forced is only recognised by the script calling the function - the key itself is not physically pressed down on the keyboard, nor is the key held down or indeed controlled in any way according to any other application.

To reset the state of the key so that it can again be controlled by the user, use either the reset_forced_key or reset_all_forced_keys function.

Example:
/*
This is a small example that plays the Windows ding every second while the space bar is pressed. If the f key is pressed, the script will force the space key down, thereby triggering the ding to play whether the space bar is physically pressed or not.
*/

void  main()
{
sound  ding;
timer  time;
ding.load("c:/windows/media/ding.wav");
show_game_window("Ding  Test");
while(true)
{
if((time.elapsed<1000)||(!key_down(KEY_SPACE)))
{
if((key_pressed(KEY_ESCAPE))||((key_down(KEY_LMENU))&&(key_pressed(KEY_F4))))
{
exit();
}
if(key_pressed(KEY_F))
{
force_key_down(KEY_SPACE);
}
wait(5);
continue;
}
time.restart();
ding.play();
}
}

force_key_up

This function causes a given key to assume the state of being released.

bool force_key_up(int key)

Parameters:
key
One of the BGT key name constants, see appendix A for a full list.

Return value:
true on success, false on failure.

Remarks:
Forcing keys is useful for controlling certain aspects of games. For example, if you wanted a game to completely replay itself in the player's style, you might record the code and length of each key pressed in a previous game, and code the replay so that all the keys are forced up or up at the right moments, thereby completely
simulating the player's actions.

The state of the key being forced is only recognised by the script calling the function and does not affect any other applications.

Please note that forcing a key up after being forced down does not pass control of the key back to the user, rather it reverses the state of the affected key. To reset the state of the key so that it can again be controlled by the user, use either the reset_forced_key or reset_all_forced_keys function.

Example:
/*
This is a small example that plays the Windows ding every second until the space bar is pressed. If the f key is pressed, the script will force the space key up, thereby triggering the ding to play whether the space bar is physically pressed or not.
*/

void  main()
{
sound  ding;
timer  time;
ding.load("c:/windows/media/ding.wav");
show_game_window("Ding  Test");
while(true)
{
if((time.elapsed<1000)||(!key_up(KEY_SPACE)))
{
if((key_pressed(KEY_ESCAPE))||((key_down(KEY_LMENU))&&(key_pressed(KEY_F4))))
{
exit();
}
if(key_pressed(KEY_F))
{
force_key_up(KEY_SPACE);
}
wait(5);
continue;
}
time.restart();
ding.play();
}
}

get_characters

This function traps any characters that are sent to the game window.

string get_characters()

Parameters:
None.

Return value:
A string containing any printable characters that have been pressed inside the game window since the last call.

Remarks:
This function only traps printable characters such as letters, numbers, punctuation, symbols, etc. All other keys such as tab,
control, escape and enter are ignored.

The function will return any characters pressed since the last call, up to 1KB, or 1024 characters. If the 1025th character is
pressed, the first character is deleted and the whole queue of data is pushed to make room for the new entry.

Example:
// Give the user five seconds to do some typing and display the result.

timer  wait_time;

void  main()
{
alert("Waiting",  "You  have  five  seconds  to  type  some  text.");
show_game_window("Typing  test");
while(true)
{
wait(5);
if(wait_time.elapsed>=5000)
{
wait_time.pause();
alert("Thanks",  "Thanks  for  waiting  for  me.  You  typed  "+get_characters()+".  Goodbye.");
exit();
}
}
}

input_box

This function displays an input box on the user's screen with an OK and a cancel button.

string input_box(string title, string text, string initial_text)

Parameters:
title
The title of the window.
text
The text that is to be shown along with the input field.
initial_text
(Optional) A string of text that should be written in the input box by default.

Return value:
The text that the user entered on success, or an empty string on failure.

Remarks:
This function is useful when you want to request some input from the user such as their name or other data. The script execution will pause until the user clicks on
either the OK or the cancel button.

The user is allowed to enter a maximum of 4096 characters, which is to say 4 kb worth of data.

Example:
//  Generate  an  input  box,  and  check  whether  the  user  pressed  cancel  by  calling  get_last_error.

void  main()
{
string  name=input_box("Name",  "Please  type  in  your  name.");
if(name=="")
{
if(get_last_error()==-12)
{
alert("OK",  "If  you  wish  to  press  cancel,  then  I  will  trouble  you  no  further.  Goodbye!");
exit();
}
alert("OK",  "If  you  wish  not  to  disclose  that  personal  precious  information,  then  I  will  trouble  you  no  further.  Goodbye!");
exit();
}
alert("Hello!",  "Hello  "+name+",  nice  to  meet  you!");
}

install_keyhook

This function installs a low level keyboard hook that forces any keys to be passed to the game window while it is in focus, regardless of any other running
applications.

bool install_keyhook()

Parameters:
None.

Return value:
true if the keyboard hook was successfully installed, false otherwise.

Remarks:
This function is useful if the game uses the Jaws for Windows screenreader interface. However it should only be used if actually required, since it will block
any global keys whilst ever the game window is focused. This can include keystrokes that allow the screenreader to perform various internal functions.

Example:
// Install a keyhook and display a message saying that the Insert key has been pressed.

void  main()
{
show_game_window("Test  Game");
bool  success=install_keyhook();
if(!success)
{
alert("Error",  "The  keyhook  could  not  be  installed.  Please  unload  the  Jaws  for  Windows  screenreader  before  playing.");
}
while(true)
{
if(key_down(KEY_LMENU)  &&  key_pressed(KEY_F4))
{
exit();
}
if(key_pressed(KEY_INSERT))
{
alert("Information",  "The  Insert  key  was  successfully  pressed.");
}
// Other code goes here.
wait(5);
}
uninstall_keyhook();
}

key_down

This function checks if a particular key on the keyboard is currently being held down.

bool key_down(int key)

Parameters:
key
One of the BGT key name constants, see appendix A for a full list.

Return value:
true if the key is held down, false if it is not or if an error occurs.

Remarks:
The difference between key_down and key_pressed is that key_pressed will only return true when the user first pushes down the
key, while key_down will continue returning true until the key is released again.

Example:
// Wait for the user to hold down alt and press f4 before closing the program.

void  main()
{
show_game_window("Test  Game");
while(true)
{
if(key_down(KEY_LMENU)  &&  key_pressed(KEY_F4))
{
exit();
}
// Other code goes here.
wait(5);
}
}

key_pressed

This function checks if a particular key on the keyboard has just been pushed down.

bool key_pressed(int key)

Parameters:
key
One of the BGT key name constants, see appendix A for a full list.

Return value:
true if the key has just been pushed down, false if it has not or if an error occurs.

Remarks:
The difference between key_down and key_pressed is that key_pressed will only return true when the user first pushes down the
key, while key_down will continue returning true until the key is released again.

Example:
// Wait for the user to hold down alt and press f4 before closing the program.

void  main()
{
show_game_window("Test  Game");
while(true)
{
if(key_down(KEY_LMENU)  &&  key_pressed(KEY_F4))
{
exit();
}
// Other code goes here.
wait(5);
}
}

key_released

This function checks if a particular key on the keyboard has just been released.

bool key_released(int key)

Parameters:
key
One of the BGT key name constants, see appendix A for a full list.

Return value:
true if the key has just been released, false if it has not or if an error occurs.

Remarks:
The difference between key_up and key_released is that key_released will only return true when the user first releases the key,
while key_up will continue returning true until the key is down.

Example:
// Speak a message when the space bar is released.

void  main()
{
tts_voice  voice;
show_game_window("Test  Game");
while(true)
{
if(key_pressed(KEY_SPACE))
{
voice.speak_interrupt("Space  bar  pressed.");
}
if(key_released(KEY_SPACE))
{
voice.speak_interrupt("Space  bar  released.");
}
if(key_pressed(KEY_ESCAPE))
{
exit();
}
// Other code goes here.
wait(5);
}
}

key_up

This function checks if a particular key on the keyboard is currently up.

bool key_up(int key)

Parameters:
key
One of the BGT key name constants, see appendix A for a full list.

Return value:
true if the key is up, false if it is not or if an error occurs.

Remarks:
The difference between key_up and key_released is that key_released will only return true when the user first releases the key,
while key_up will continue returning true until the key is pushed down again.

Example:
// Time how long the space bar is up.

void  main()
{
timer  up;
show_game_window("keyboard  Test");
up.restart();
while(!key_pressed(KEY_ESCAPE))
{
if(key_up(KEY_SPACE))
{
continue;
}
alert("Time",  "The  space  bar  has  been  up  for  "+(up.elapsed/1000)+"  seconds.");
}
}
}

keys_down

This function returns a list of all the keys on the keyboard that are currently being held down.

int[] keys_down()

Parameters:
None.

Return value:
An array with the currently held keys, or an empty array if no keys are being held down or if an error occurs.

Remarks:
The difference between keys_down and keys_pressed is that keys_pressed will only return a list of the keys that the user has just
pushed down, while keys_down will return a list of the keys that are currently being held down regardless of whether the same
keys have been reported as being down in a previous call.

Example:
// Wait for three different keys to be held down at once before closing the program.

void  main()
{
show_game_window("Keys  Down  Test");
int[]  key_list;
while(key_list.length()<3)
{
key_list=keys_down();
wait(5);
}
}

keys_pressed

This function returns a list of all the keys on the keyboard that have just been pressed.

int[] keys_pressed()

Parameters:
None.

Return value:
An array with the keys that have just been pressed, or an empty array if no keys have been pressed since the last call or if an
error occurs.

Remarks:
The difference between keys_down and keys_pressed is that keys_pressed will only return a list of the keys that the user has just
pushed down, while keys_down will return a list of the keys that are currently being held down regardless of whether the same
keys have been reported as being down in a previous call.

Please note that keys_pressed and key_pressed use the same keyboard state, so if a call to keys_pressed is made right before a
call to key_pressed, none of the keys returned by keys_pressed will be reported as just having been pushed down by
key_pressed. The same holds true for the reverse scenario.

Example:
// Speak the key codes of any keys that are pressed, and exit if the user presses escape.

void  main()
{
show_game_window("Keys  Pressed  Test");
tts_voice  speech;
int[]  list;
while(key_pressed(KEY_ESCAPE)==false)
{
list=keys_pressed();
for(uint  i=0;i<list.length();i++)
{
if(i==0)
{
speech.speak_interrupt(list[i]);
}
else
{
speech.speak(list[i]);
}
}
wait(5);
}
}

keys_released

This function returns a list of all the keys on the keyboard that have just been released.

int[] keys_released()

Parameters:
None.

Return value:
An array with the keys that have just been released, or an empty array if no keys have been released since the last call or if an
error occurs.

Remarks:
The difference between keys_up and keys_released is that keys_released will only return a list of the keys that the user has just
released, while keys_up will return a list of the keys that are currently up regardless of whether the same keys have been
reported as being up in a previous call.

Please note that keys_released and key_released use the same keyboard state, so if a call to keys_released is made right before
a call to key_released, none of the keys returned by keys_released will be reported as just having been released by
key_released. The same holds true for the reverse scenario.

Example:
// Speak the key codes of any keys that are released, and exit if the user presses escape.

void  main()
{
show_game_window("Keys  released  Test");
tts_voice  speech;
int[]  list;
while(key_pressed(KEY_ESCAPE)==false)
{
list=keys_released();
for(uint  i=0;i<list.length();i++)
{
if(i==0)
{
speech.speak_interrupt(list[i]);
}
else
{
speech.speak(list[i]);
}
}
wait(5);
}
}

keys_up

This function returns a list of all the keys on the keyboard that are currently up.

int[] keys_up()

Parameters:
None.

Return value:
An array with the keys that are up, or an empty array if no keys are up or if an error occurs.

Remarks:
The difference between keys_up and keys_released is that keys_released will only return a list of the keys that the user has just released, while keys_up will return
a list of the keys that are currently up regardless of whether the same keys have been reported as being up in a previous call.

Example:
// The example will constantly monitor keys that are up. If it registers three or more keys that are not up, it will exit.

void  main()
{
show_game_window("Keys  up  Test");
int[]  key_list;
while(key_list.length()>253)
{
key_list=keys_up();
wait(5);
}
}

mouse_down

This function checks if a particular mouse button is currently being held down.

bool mouse_down(int button)

Parameters:
button
A mouse button number from 0 to 7 (see remarks).

Return value:
true if the button is held down, false if it is not or if an error occurs.

Remarks:
The difference between mouse_down and mouse_pressed is that mouse_pressed will only return true when the user first pushes
down the button, while mouse_down will continue returning true until the button is released again.

The button number can be between 0 and 7, however not many mouse devices will have as many as 8 buttons. The numbers
correspond to the following buttons:

0 - Left button.
1 - Right button.
2 - Middle button.
3 - Button 4.
4 - Button 5.
5 - Button 6.
6 - Button 7.
7 - Button 8.

Example:
// Play a looping sound while the left mouse button is held down, and use escape to close.

void  main()
{
show_game_window("Test  Game");
sound  horn;
horn.load("horn.wav");
if(horn.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
while(true)
{
mouse_update();
if(mouse_down(0))
{
if(horn.playing==false)
{
horn.play_looped();
}
}
else
{
if(horn.playing==true)
{
horn.stop();
}
}
if(key_pressed(KEY_ESCAPE))

{
exit();
}
wait(5);
}
}

mouse_pressed

This function checks if a particular mouse button has just been pushed down.

bool mouse_pressed(int button)

Parameters:
button
A mouse button number from 0 to 7 (see remarks).

Return value:
true if the button has just been pushed down, false if it has not or if an error occurs.

Remarks:
The difference between mouse_down and mouse_pressed is that mouse_pressed will only return true when the user first pushes
down the button, while mouse_down will continue returning true until the button is released again.

The button number can be between 0 and 7, however not many mouse devices will have as many as 8 buttons. The numbers
correspond to the following buttons:

0 - Left button.
1 - Right button.
2 - Middle button.
3 - Button 4.
4 - Button 5.
5 - Button 6.
6 - Button 7.
7 - Button 8.

Example:
// Wait for the user to press the left mouse button or the escape key before closing the program.

void  main()
{
show_game_window("Test  Game");
while(true)
{
mouse_update();
if(mouse_pressed(0)  or  key_pressed(KEY_ESCAPE))
{
exit();
}
// Other code goes here.
wait(5);
}
}

mouse_update

This function updates the mouse device to see if any movement or button presses have occurred.

bool mouse_update()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This function updates the state of the mouse device in order to retrieve any changes that have occurred since the last call. It is important that you call this function very frequently in order for the mouse movement to be responsive and accurate.

The mouse is a relative movement device, which means that there is no meaningful absolute position information that can be retrieved. Instead, each call to this function updates the three global mouse movement constants with the amount that the mouse has moved in each direction since the
last call. Before the first call to this function, all three constants will be 0. Similarly, if no movement has been made in a particular direction since the last call, the constant in question will also be 0.

The three constants that may be accessed are MOUSE_X, MOUSE_Y and MOUSE_Z. MOUSE_X and MOUSE_Y represent the left and right and backward and forward movement respectively, while MOUSE_Z indicates the movement of the wheel if one is present.

You may also use the mouse_down and mouse_pressed functions to check the state of the available mouse buttons.

Please note that in order to avoid a situation where a mouse click may accidentally move focus away from the game window, BGT acquires exclusive access to the device. This in turn means that the visible cursor disappears and that the mouse cannot be used to get out of the window. For this
reason, the mouse is only acquired when it is actually being called upon by the script writer. It is acquired when mouse_update is called for the first time, and automatically released again if mouse_update is not called for one second. Should mouse_update be called again at a later point, the
mouse will be reacquired. However if you use the mouse in your game, be sure to call mouse_update very frequently but only when you are actually using it as to allow users to leave the window with a click as often as possible during the games lifetime.

Example:
//  This  is  a  simple  and  rather  unrealistic  truck  simulation.

/*
This program plays an engine sound and allows the user to move it around with the mouse. The code lowers the sensitivity of the mouse slightly in order to make changes less rapid. Escape is pressed to close the program.
*/

void  main()
{
show_game_window("Mouse  Truck");
sound  truck;
truck.load("engine.wav");
if(truck.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
double  current_pitch=20;
double  current_pan=0;
truck.pitch=current_pitch;
truck.play_looped();
while(key_pressed(KEY_ESCAPE)==false)
{
mouse_update();

// Now that we've updated the mouse, we get the movement as doubles so we can manipulate it easier.
double  x=MOUSE_X;
double  y=MOUSE_Y;

// Has the x position changed?
if(x<0.0  or  x>0.0)
{

// It has, so recalculate the movement amount to be less sensitive.
x/=1.5;

current_pan+=x;
if(current_pan<-60)
{
current_pan=-60;
}
if(current_pan>60)
{
current_pan=60;
}
truck.pan=current_pan;
}

// Has the y position changed?
if(y<0.0  or  y>0.0)
{

// It has, so recalculate the movement amount to be less sensitive.
y/=1.5;
current_pitch-=y;
if(current_pitch<20)
{
current_pitch=20;
}
if(current_pitch>200)
{
current_pitch=200;
}
truck.pitch=current_pitch;
}
wait(5);
}
}

question

This function displays a message box on the user's screen with a title and text, as well as a yes and a no button.

double question(string title, string text)

Parameters:
title
The title of the window.
text
The text that is to be shown.

Return value:
1 for yes, 2 for no, or 0 on failure.

Remarks:
This function is useful when you want to ask the user a simple question. The script execution will pause until the user clicks on
either the yes or the no button.

Example:
void  main()
{
int  answer  =  question("Question",  "Would  you  like  to  open  my  fruit  basket?");
if(answer==0)
{
alert("Wait!",  "I'm  afraid  something  went  horribly  wrong  here!");
}
if(answer==1)
{
alert("Great!",  "I'll  show  you  in  a  minute,  just  hang  on...");
}
if(answer==2)
{
alert("Oh well", "I sure don't mind, all the more for me to consume on my own.");
}
}

reset_all_forced_keys

This function causes all keys to reset their state of being forced to pass control back to the user.

bool reset_all_forced_keys()

Parameters:
None.

Return value:
true if the keys' states are reset successfully, false if an error occurs.

Remarks:
Forcing keys is useful for controlling certain aspects of games. For example, if you wanted a game to completely replay itself in the player's style, you might record the code and length of each key pressed in a previous game, and code the replay so that all the keys are forced up or down at the right moments, thereby completely simulating the player's actions.

This function should be called after the use of either the force_key_down or force_key_up function. To reset the state of any key individually, use the reset_forced_key function.

Example:
/*
This is a small example that plays the Windows ding every second while the space bar is pressed. If the f key is pressed, the script will force the space key down, thereby triggering the ding to play whether the space bar is physically pressed or not. Pressing the r key will reset the state.
*/

void  main()
{
sound  ding;
timer  time;
ding.load("c:/windows/media/ding.wav");
show_game_window("Ding  Test");
while(true)
{
if((time.elapsed<1000)||(!key_down(KEY_SPACE)))
{
if((key_pressed(KEY_ESCAPE))||((key_down(KEY_LMENU))&&(key_pressed(KEY_F4))))
{
exit();
}
if(key_pressed(KEY_F))
{
force_key_down(KEY_SPACE);
}
if(key_pressed(KEY_R))
{
reset_all_forced_keys();
}
wait(5);
continue;
}
time.restart();
ding.play();
}
}

reset_forced_key

This function causes a given key to reset its state of being forced to pass control back to the user.

bool reset_forced_key(int key)

Parameters:
key
One of the BGT key name constants, see appendix A for a full list.

Return value:
true if the key's state is reset successfully, false if an error occurs.

Remarks:
Forcing keys is useful for controlling certain aspects of games. For example, if you wanted a game to completely replay itself in the player's style, you might record the code and length of each key pressed in a previous game, and code the replay so that all the keys are forced up or down at the right moments, thereby completely simulating the player's actions.

This function should be called after the use of either the force_key_down or force_key_up function. To reset the state of all keys, use the reset_all_forced_keys function.

Example:
/*
This is a small example that plays the Windows ding every second while the space bar is pressed. If the f key is pressed, the script will force the space key down, thereby triggering the ding to play whether the space bar is physically pressed or not. Pressing the r key will reset the state.
*/

void  main()
{
sound  ding;
timer  time;
ding.load("c:/windows/media/ding.wav");
show_game_window("Ding  Test");
while(true)
{
if((time.elapsed<1000)||(!key_down(KEY_SPACE)))
{
if((key_pressed(KEY_ESCAPE))||((key_down(KEY_LMENU))&&(key_pressed(KEY_F4))))
{
exit();
}
if(key_pressed(KEY_F))
{
force_key_down(KEY_SPACE);
}
if(key_pressed(KEY_R))
{
reset_forced_key(KEY_SPACE);
}
wait(5);
continue;
}
time.restart();
ding.play();
}
}

screen_reader_is_running

This function checks to see if the given screen reader is active.

bool screen_reader_is_running(int reader)

Parameters:
reader
A value representing one of the four supported screen readers (see remarks).

Return value:
true if the given screen reader is running, false if it is not or if an error occurs.

Remarks:
This function supports four available screen readers, each of which are identified by a number, as follows:

JAWS (1) - Jaws for Windows (Freedom Scientific)
WINDOW_EYES (2) - Window Eyes (GWMicro)
SYSTEM_ACCESS (3) - System Access (Serotek)
NVDA (4) - Non Visual Desktop Access (NVDA Project)

Please note that all dependencies must be correctly installed in order for this function to work correctly. If this is not the case, the function will return false, even if the screen reader may in fact be
active. See Appendix F for information about dependencies.

Example:
// Check if any screen readers are running and get that screen reader to speak some text. If no screen reader is active, we will display an alert box.

void  main()
{
bool  found_reader=false;
for(int  counter=1;  counter<5;  counter++)
{
if(screen_reader_is_running(counter))
{
found_reader=true;
screen_reader_speak(counter,  "Hi.  Bet  ya  didn't  know  I  could  say  things  on  my  own  without  a  screen,  did  you?");
break;
}
}
if(!found_reader)
{
alert("Information",  "I  can't  seem  to  communicate  with  any  screen  readers,  so  I'll  just  have  to  show  you  this  instead.");
}
}

screen_reader_set_library_path

This function will set the path and filename of the dll library for a given screen reader.

bool screen_reader_set_library_path(int reader, string filename)

Parameters:
reader
A value representing one of the two screen readers for which this operation is supported (see remarks).
filename
The filename of the DLL.

Return value:
true on success, false on failure.

Remarks:
This function expects a complete path and filename of an existing file, or it will fail. The name must be included though the
extension is not important, it doesn't have to be .dll.

If this function is not called the engine will assume the libraries to be in the same directory as your script, and that they have the
default filenames as set by the manufacturer.

This function will work with two of the four supported screen readers, namely System Access and NVDA. If you attempt to use
this function with Jaws or WindowEyes, the function will fail as these readers use COM rather than standard DLL calls.

When you call this function, if the screen reader library is already loaded it will be unloaded automatically before the path is
changed. The next call to any of the other screen reader functions except screen_reader_unload_library will then automatically
initialize the screen reader library with your new path.

Example:
// Set the path of the NVDA screen reader and get it to speak some text.

void  main()
{
if(!screen_reader_set_library_path(NVDA,  DIRECTORY_TEMP+"/NVDA.dll"))
{
alert("Error",  "There  was  an  error  interacting  with  the  NVDA  screen  reader.");
exit();
}
screen_reader_speak(NVDA,  "Hi.  My  library  is  now  in  the  temp  directory!");
}

screen_reader_speak

This function speaks some text using a screen reader.

bool screen_reader_speak(int reader, string text)

Parameters:
reader
A value representing one of the four supported screen readers (see remarks).
text
The text that is to be spoken.

Return value:
true on success, false on failure.

Remarks:
This function supports four available screen readers, each of which are identified by a number, as follows:

JAWS (1) - Jaws for Windows (Freedom Scientific)
WINDOW_EYES (2) - Window Eyes (GWMicro)
SYSTEM_ACCESS (3) - System Access (Serotek)
NVDA (4) - Non Visual Desktop Access (NVDA Project)

Please note that this function does not print anything on screen, merely uses an internal API provided by the screen reader in order to speak the
text.
Please note that the screen reader must be running and all dependencies correctly installed in order for this function to work correctly. If these
steps are not met, the function will return false and no text will be spoken. See Appendix F for information about dependencies.

Example:
// Get Jaws to speak some text.

void  main()
{
screen_reader_speak(JAWS, "Hi. Bet ya didn't know I could say things on my own without a screen, did you?");
}

screen_reader_speak_interrupt

This function speaks some text using a screen reader, interrupting any information that it is currently reading.

bool screen_reader_speak_interrupt(int reader, string text)

Parameters:
reader
A value representing one of the four supported screen readers (see remarks).
text
The text that is to be spoken.

Return value:
true on success, false on failure.

Remarks:
This function supports four available screen readers, each of which are identified by a number, as follows:

JAWS (1) - Jaws for Windows (Freedom Scientific)
WINDOW_EYES (2) - Window Eyes (GWMicro)
SYSTEM_ACCESS (3) - System Access (Serotek)
NVDA (4) - Non Visual Desktop Access (NVDA Project)

Please note that this function does not print anything on screen, merely uses an internal API provided by the screen reader in order to speak the text.
Please note that the screen reader must be running and all dependencies correctly installed in order for this function to work correctly. If these steps are not
met, the function will return false and no text will be spoken. See Appendix F for information about dependencies.

Example:
// Get Jaws to speak some text.

void  main()
{
screen_reader_speak_interrupt(JAWS,  "Hi.  Bet  ya  didn't  know  I  could  say  things  on  my  own  without  a  screen,  did  you?");
}

screen_reader_stop_speech

This function will stop any information that is currently being spoken by a screen reader.

bool screen_reader_stop_speech(int reader)

Parameters:
reader
A value representing one of the four supported screen readers (see remarks).

Return value:
true on success, false on failure.

Remarks:
This function supports four available screen readers, each of which are identified by a number, as follows:

JAWS (1) - Jaws for Windows (Freedom Scientific)
WINDOW_EYES (2) - Window Eyes (GWMicro)
SYSTEM_ACCESS (3) - System Access (Serotek)
NVDA (4) - Non Visual Desktop Access (NVDA Project)

Please note that the screen reader must be running and all dependencies correctly installed in order for this function to work correctly. If these steps are not met, the function will return false and no action will be taken. See Appendix F for information about dependencies.

Example:
// Get Jaws to speak some text and stop it after 2 seconds.

void  main()
{
screen_reader_speak(JAWS, "Hi. Bet ya didn't know I could say things on my own without a screen, did you? Well, I'm speaking a fairly long string, and if I haven't stopped by now, there's something definitely wrong!");
wait(2000);
screen_reader_stop_speech(JAWS);
}

screen_reader_unload_library

This function will unload the DLL for the specified screen reader.

bool screen_reader_unload_library(int reader)

Parameters:
reader
A value representing one of the two screen readers for which this operation is supported (see remarks).

Return value:
true on success, false on failure.

Remarks:
While Windows is using a dll, it cannot be deleted or removed. Therefore this function is useful if you want to delete the DLL file, for
instance if you have packaged it into an executable and extract it to a temporary location during the execution of your script. After
unloading the dll for the given screen reader, it may then be manipulated like any other file.

This function will work with two of the four supported screen readers, namely System Access and NVDA. If you attempt to use this
function with Jaws or WindowEyes, the function will fail as these readers use COM rather than standard DLL calls.

Note that if you call any screen reader functions later except screen_reader_set_library_path, it will automatically attempt to reload the dll
for that screen reader if applicable.

Example:
// Set the path of the NVDA screen reader and get it to speak some text, then delete the DLL afterwards.

void  main()
{
screen_reader_set_library_path(NVDA,  DIRECTORY_TEMP+"/NVDA.dll");
screen_reader_speak(NVDA,  "Hi.  My  library  is  now  in  the  temp  directory!");
wait(2000);
screen_reader_unload_library(NVDA);
file_delete(DIRECTORY_TEMP+"/NVDA.dll");
}

uninstall_keyhook

This function uninstalls a low level keyboard hook previously installed with the install_keyhook function.

bool uninstall_keyhook()

Parameters:
None.

Return value:
true if the keyboard hook was successfully uninstalled, false otherwise.

Remarks:
The program will automatically uninstall a keyhook on exit.

Example:
// Install a keyhook and display a message saying that the Insert key has been pressed.

void  main()
{
show_game_window("Test  Game");
install_keyhook();
while(true)
{
if(key_down(KEY_LMENU)  &&  key_pressed(KEY_F4))
{
exit();
}
if(key_pressed(KEY_INSERT))
{
alert("Information",  "The  Insert  key  was  successfully  pressed.");
}
// Other code goes here.
wait(5);
}
uninstall_keyhook();
}

ascii_to_character

This function returns the character that corresponds to a given ASCII code.

string ascii_to_character(int the_ascii_code)

Parameters:
the_ascii_code
The ASCII code that will be converted.

Return value:
A string containing the corresponding character, or an empty string on error.

Remarks:
This function supports all ASCII values from 0 to 255, which includes the extended character set. This means that foreign
characters and currency symbols can also be converted.

If the supplied ASCII code is out of range, an empty string will be returned.

Example:
void  main()
{
alert("ascii_to_character  test",  ascii_to_character(65));  //  Output  should  be  A
}

character_to_ascii

This function returns the ASCII code of a character.

int character_to_ascii(string the_character)

Parameters:
the_character
The character that will be converted.

Return value:
An integer containing the given character's ASCII code, or -1 on error.

Remarks:
If an empty string is passed, the function will return -1.

This function supports all ASCII values from 0 to 255, which includes the extended character set. This means that foreign
characters and currency symbols can also be converted.

If more than one character is entered into the string parameter, the function converts the first character and ignores the rest.

Example:
void  main()
{
alert("character_to_ascii  test",  character_to_ascii("A"));  //  Output  should  be  65
}

hex_to_string

This function converts a string of hexadecimal numbers into standard text.

string hex_to_string(string the_hexadecimal_sequence)

Parameters:
the_hexadecimal_sequence
The string of hexadecimal numbers that will be converted.

Return value:
A string containing the standard text equivalent of the given hexadecimal data, or an empty string on error.

Remarks:
Hexadecimal numbers range from 0 to 9, and then a (or A) to f (or F) to represent numbers 10 to 15. If other characters are used then a blank
string is returned.

The converted string will be half the length of the given data.

Example:
void  main()
{
alert("hex_to_string  test",  hex_to_string("412066726F6720697320626F756E63696E6720696E206D7920666F6F6421"));
}

number_to_hex_string

This function converts a number to its corresponding hexadecimal string representation.

string number_to_hex_string(double the_number)

Parameters:
the_number
The number to convert.

Return value:
A string containing the hexadecimal representation of the number, or an empty string on error.

Remarks:
This function only works with positive numbers, without decimals. If the number contains decimals, these will simply be removed
without rounding.

Please note that the returned string uses lowercase letters. If you want the result to be compatible with the hex_to_string function
for example, it is necessary to call string_to_upper_case on it first.

Example:
void  main()
{
alert("number_to_hex_string  test",  number_to_hex_string(1000));  //  Output  should  be  3e8.
}

number_to_words

This function converts an integral number into English words.

string number_to_words(double number, bool include_and)

Parameters:
number
The number to convert.
include_and
A boolean which specifies whether or not the word "and" should be inserted in the appropriate places in the output string.

Return value:
The word representation of the number on success, or an empty string on failure.

Remarks:
This function only works with numbers (positive or negative), without decimals. If the supplied number contains decimals, these
will simply be removed without rounding.

If the number is lower than -999999999 (minus nine hundred ninety nine million nine hundred ninety nine thousand nine hundred
ninety nine) or greater than 999999999 (nine hundred ninety nine million nine hundred ninety nine thousand nine hundred ninety
nine), the function will fail.

This function is very useful when you wish to have a number spoken by using concatenated sound files, for instance. Simply split
the words up into an array with the string_split function, and then loop through the resulting array and play an appropriate sound
file for each word.

Example:
void  main()
{
long  my_number=random(-1000,  1000);
string  result=number_to_words(my_number,  true);
alert("Result", my_number + " is written as " + result + ".");
}

string_base64_decode

This function returns a string decoded from its base64 representation.

string string_base64_decode(string base64_string)

Parameters:
base64_string
The base64 string that will be decoded.

Return value:
A string containing the decoded content on success, or an empty string on error.

Remarks:
Base64 is a way in which to represent binary data in a textual format. This means that the NULL character, ASCII code 0, will
never be present in a base64 encoded string and it may therefore be used to store binary data in situations where only text
content is supported. Converting binary data to a hexadecimal representation accomplishes the same thing, but base64 encoded
strings are shorter than hexadecimal strings and are thus more suitable when space is an issue.

Example:
void  main()
{
string  original=input_box("Enter  string",  "Enter  the  string  to  convert  to  base64.");
alert("Original",  original);
string  encoded=string_base64_encode(original);
alert("Encoded",  encoded);
string  decoded=string_base64_decode(encoded);
alert("Decoded",  decoded);
}

string_base64_encode

This function returns the base64 representation of a string.

string string_base64_encode(string the_string)

Parameters:
the_string
The string that will be encoded.

Return value:
A string containing the base64 representation of the given string on success, or an empty string on error.

Remarks:
Base64 is a way in which to represent binary data in a textual format. This means that the NULL character, ASCII code 0, will
never be present in a base64 encoded string and it may therefore be used to store binary data in situations where only text
content is supported. Converting binary data to a hexadecimal representation accomplishes the same thing, but base64 encoded
strings are shorter than hexadecimal strings and are thus more suitable when space is an issue.

Example:
void  main()
{
string  original=input_box("Enter  string",  "Enter  the  string  to  convert  to  base64.");
alert("Original",  original);
string  encoded=string_base64_encode(original);
alert("Encoded",  encoded);
string  decoded=string_base64_decode(encoded);
alert("Decoded",  decoded);
}

string_contains

This function searches a string for a specific text and, if found, returns the position where the text starts.

int string_contains(string the_string, string the_search, int occurrence)

Parameters:
the_string
The string that will be searched.
the_search
The text to search for within the string.
occurrence
The occurrence of the text to return (starting at 1).

Return value:
The position in the source string where the search string was found on success, or -1 if there was an error or the text could not be found.

Remarks:
Please note that the position returned is 0 based.

The search is case sensitive, meaning that Name is different from name, for instance.

Example:
void  main()
{
alert("string_contains  test  1",  "string_contains="+string_contains("Hello,  I  am  a  string!",  "ring",  1));  //output  should  be  16.
alert("string_contains  test  2",  "string_contains="+string_contains("Hello,  I  will  bring  you  a  string!",  "ring",  2));  //output  should  be  28.
alert("string_contains  test  3",  "string_contains="+string_contains("Hello,  I  am  a  string!",  "Ring",  1));  //output  should  be  -1,  as  Ring  is  different  from  ring.
}

string_distance

This function calculates the syntactical similarity between two strings.

int string_distance(string first_string, string second_string)

Parameters:
first_string
The first string.
second_string
The second string.

Return value:
The distance between the two strings on success (see remarks), or -1 on failure.

Remarks:
This function uses what is known as the Levenshtein distance algorithm to measure the similarity between two arbitrary strings, by calculating how many changes
that would be necessary to turn the one string into the other. The number of required operations thus becomes the similarity between the two strings. The similarity
between the strings "man" and "mandy" will be 2, for instance, since all that is required in order to turn man into mandy is adding a d and a y to the end of the
former. Similarly, turning car into bar requires the substitution of c to b which is considered to be one operation, and as a result the similarity is 1. The three
permissible operations are addition, subtraction (e.g. removing a character), and substitution (which is to say replacing one character with another).

Please note that this function uses a considerable amount of memory, and should thus not be called on longer strings.

Example:
void  main()
{
alert("Similarity",  "The  similarity  between  the  words  \"man\"  and  \"mandy\"  is  "  +  string_distance("man",  "mandy")  +  ".");
}

string_is_alphabetic

This function returns a boolean value showing whether the passed string contains purely letters.

bool string_is_alphabetic(string the_string)

Parameters:
the_string
The string that will be evaluated.

Return value:
Returns true if the string contains purely letters, false if not or if an error occurred.

Remarks:
This function does not evaluate punctuation. If the string contains punctuation this function will return false.

Example:
void  main()
{
string  text1,  text2;
text1="I  am  normal  text.";
text2="Randomness";
if(string_is_alphabetic(text1))
{
alert("string_is_alphabetic  test",  "text1  contains  letters  only.");
}
else
{
alert("string_is_alphabetic  test",  "text1  does  not  contain  only  letters.");
}
if(string_is_alphabetic(text2))
{
alert("string_is_alphabetic  test",  "text2  contains  letters  only.");
}
else
{
alert("string_is_alphabetic  test",  "text2  does  not  contain  only  letters.");
}
}

string_is_alphanumeric

This function returns a boolean value showing whether the passed string contains letters and numbers.

bool string_is_alphanumeric(string the_string)

Parameters:
the_string
The string that will be evaluated.

Return value:
Returns true if the string contains letters and/or numbers, false if not or if an error occurred.

Remarks:
This function does not evaluate punctuation. If the string contains punctuation this function will return false.

Example:
void  main()
{
string  text1,  text2;
text1="I  am  normal  text.";
text2="Randomness15";
if(string_is_alphanumeric(text1))
{
alert("string_is_alphanumeric  test",  "text1  contains  letters  and  numbers.");
}
else
{
alert("string_is_alphanumeric  test",  "text1  does  not  contain  only  letters  and  numbers.");
}
if(string_is_alphanumeric(text2))
{
alert("string_is_alphanumeric  test",  "text2  contains  letters  and  numbers.");
}
else
{
alert("string_is_alphanumeric  test",  "text2  does  not  contain  only  letters  and  numbers.");
}
}

string_is_digits

This function returns a boolean value showing whether the passed string contains purely digits.

bool string_is_digits(string the_string)

Parameters:
the_string
The string that will be evaluated.

Return value:
Returns true if the string contains purely digits, false if not or if an error occurred.

Remarks:
This function does not evaluate numbers, but rather digits. If the string contains a negative or floating point number this function
will return false.

Example:
void  main()
{
string  text1,  text2;
text1="0.5";
text2="15244";
if(string_is_digits(text1))
{
alert("string_is_digits  test",  "text1  contains  digits  only.");
}
else
{
alert("string_is_digits  test",  "text1  does  not  contain  only  digits.");
}
if(string_is_digits(text2))
{
alert("string_is_digits  test",  "text2  contains  digits  only.");
}
else
{
alert("string_is_digits  test",  "text2  does  not  contain  only  digits.");
}
}

string_is_lower_case

This function returns a boolean value showing whether the passed string contains purely lower case letters.

bool string_is_lower_case(string the_string)

Parameters:
the_string
The string that will be evaluated.

Return value:
Returns true if the string contains purely lower case letters, false if not or if an error occurred.

Remarks:
This function does not evaluate punctuation. If the string contains punctuation this function will return false.

Example:
void  main()
{
string  text1,  text2;
text1="I  am  normal  text.";
text2="randomness";
if(string_is_lower_case(text1))
{
alert("string_is_lower_case  test",  "text1  contains  lower  case  letters  only.");
}
else
{
alert("string_is_lower_case  test",  "text1  does  not  contain  only  lower  case  letters.");
}
if(string_is_lower_case(text2))
{
alert("string_is_lower_case  test",  "text2  contains  lower  case  letters  only.");
}
else
{
alert("string_is_lower_case  test",  "text2  does  not  contain  only  lower  case  letters.");
}
}

string_is_upper_case

This function returns a boolean value showing whether the passed string contains purely upper case letters.

bool string_is_upper_case(string the_string)

Parameters:
the_string
The string that will be evaluated.

Return value:
Returns true if the string contains purely upper case letters, false if not or if an error occurred.

Remarks:
This function does not evaluate punctuation. If the string contains punctuation this function will return false.

Example:
void  main()
{
string  text1,  text2;
text1="I  am  normal  text.";
text2="RANDOMNESS";
if(string_is_upper_case(text1))
{
alert("string_is_upper_case  test",  "text1  contains  upper  case  letters  only.");
}
else
{
alert("string_is_upper_case  test",  "text1  does  not  contain  only  upper  case  letters.");
}
if(string_is_upper_case(text2))
{
alert("string_is_upper_case  test",  "text2  contains  upper  case  letters  only.");
}
else
{
alert("string_is_upper_case  test",  "text2  does  not  contain  only  upper  case  letters.");
}
}

string_left

This function returns a specified number of characters from the left hand side of a string.

string string_left(string the_string, uint count)

Parameters:
the_string
The string that will be extracted from.
count
The number of characters to return.

Return value:
A string containing the leftmost count characters of the string.

Remarks:
If count is less than 1, a blank string will be returned. Likewise, if count is greater than the string's length, then the whole string
will be returned.

Example:
void  main()
{
alert("string_left  test",  string_left("Hello,  I  am  a  string!",  5));  //output  should  be  Hello
}

string_len

This function returns the length of the string.

double string_len(string the_string)

Parameters:
the_string
The string that will be evaluated.

Return value:
A double with the length of the string.

Remarks:
If an empty string is passed, the function will return 0.

An alternative method to string_len is to call the length method on the string variable in question, remembering that a string is an
array of characters. This function has been implemented for convenience and completeness.

Example:
void  main()
{
alert("string_len  test",  string_len("Hello,  I  am  a  string!"));  //output  should  be  21
}

string_mid

This function returns a specified number of characters from any given position of a string.

string string_mid(string the_string, uint start_position, uint count)

Parameters:
the_string
The string that will be extracted from.
start_position
The position to start extracting from.
count
The number of characters to return.

Return value:
The extracted string.

Remarks:
If the count is less than 1, a blank string will be returned, regardless of the position value.
If the count is greater than the length of the string minus the start position, the remainder of the string is returned.
If start_position is less than 1, the function will extract from the left hand side of the string.
If start_position is greater than the length of the string, a blank string is returned.

Example:
void  main()
{
alert("string_mid  test",  string_mid("Hello,  I  am  here  to  test  the  string_mid  function!",  30,  10));  //output  should  be  string_mid
}

string_replace

This function searches and replaces a specific text within a string and returns the result.

string string_replace(string the_string, string the_search, string replacement, bool replace_all)

Parameters:
the_string
The string that will be processed.
the_search
The text to search for within the string.
replacement
The string that will replace an occurrence of the search string.
replace_all
A boolean flag indicating whether every occurrence of the search should be replaced. If set to false, only the first occurrence is replaced.

Return value:
A new string containing any specified replacements. If no characters were replaced, or the search string was blank, the original string is returned. If the main text string is blank, a blank string is returned.

Remarks:
If the replacement string is blank, this has the effect of removing the search text from the string.

The search is case sensitive, meaning that Name is different from name, for instance.

Example:
void  main()
{
alert("string_replace  test",  "string_replace="+string_replace("Hello,  I  am  a  strange  string!",  "string",  "strange",  true));  //output  should  be  Hello,  I  am  a  strange  strange!
}

string_reverse

This function reverses a given string.

string string_reverse(string the_string)

Parameters:
the_string
The string that will be reversed.

Return value:
The string in reverse.

Remarks:
If an empty string is passed then no string will be returned.

Example:
void  main()
{
alert("string_reverse  test",  string_reverse("Hello,  I  am  a  string!"));  //output  should  be  !gnirts  a  ma  I  ,olleH
}

string_right

This function returns a specified number of characters from the right hand side of a string.

string string_right(string the_string, uint count)

Parameters:
the_string
The string that will be extracted from.
count
The number of characters to return.

Return value:
A string containing the rightmost count characters of the string.

Remarks:
If count is less than 1, a blank string will be returned. Likewise, if count is greater than the string's length, then the whole string
will be returned.

Example:
void  main()
{
alert("string_right  test",  string_right("Hello,  I  am  a  string!",  7));  //output  should  be  string!
}

string_split

This function splits a string into an array based on a delimiter.

string[] string_split(string source, string delimiter, bool require_full_delimiter)

Parameters:
source
The string that is to be split.
delimiter
A string containning one or more characters that, when found in the source string, cause a new array entry to be made.
require_full_delimiter
A boolean value (either true or false) that decides whether the entire sequence of characters in the delimiter needs to be present in the source string for
a new array entry to be made (see remarks).

Return value:
An array with the result of the splitting operation. If no splits took place, an array with one element containing the entire source string is returned.

Remarks:
This function is useful when dividing content, such as splitting up a longer text into individual words or sentenses. It returns an array containning the
individual strings that were present in between the delimiters. Thus, the delimiters themselves are not stored in the array; only the content that remains if
any.

If the require_full_delimiter boolean is set to true, the entire delimiter string has to be found in the source string in order for a new array entry to be
made. If it is set to false, a new array entry will be made as soon as any one of the characters in the delimiter string is found. Note however that
multiple, empty entries will not be made if one or more delimiters appear after one another; you will always have data in every entry.

Example:
// Split up sentenses into individual words using space, . , ! and ? as the delimiters.

void  main()
{
string  my_string="This  is  a  lengthy  string,  and  its  only  purpose  is  to  demonstrate  splitting.  Understand?  Good!";
alert("String  to  split",  my_string);
string[]  output_array=string_split(my_string,  "  .,!?",  false);
alert("Found  words",  output_array.length()  +  "  word(s)  were  in  the  string.");
for(uint  i=0;i  <  output_array.length();i++)
{
alert("Word  "  +  (i+1)  +  "",  output_array[i]);
}
}

string_to_hex

This function returns the hexadecimal equivalent of a string.

string string_to_hex(string the_string)

Parameters:
the_string
The string that will be converted.

Return value:
A string containing the hexadecimal equivalent of the given string, or an empty string on error.

Remarks:
Given the nature of a hexadecimal string, the converted string will be twice the length of the source string. Therefore care must be
taken if converting considerably large strings.

Example:
void  main()
{
alert("string_to_hex  test",  string_to_hex("A  frog  is  bouncing  in  my  food!"));
}

string_to_lower_case

This function converts all upper case characters in a string to lower case.

string string_to_lower_case(string the_string)

Parameters:
the_string
The string that will be processed.

Return value:
Returns a string with all upper case letters converted to lower case, or the original string on failure.

Remarks:
All punctuation and lower case characters present in the source string will be kept in tact.

Example:
void  main()
{
alert("String_to_lower_case  test",  string_to_lower_case("I  will  be  converted  to  lower  case."));
}

string_to_number

This function converts the given string into a number.

double string_to_number(string the_string)

Parameters:
the_string
The string to convert.

Return value:
The number stored in the string on success, 0 on failure.

Remarks:
When converting a string to a number the function will ignore any spaces that may appear before the actual number, as well as
any additional text that may come after the last numeric character that can be interpreted. The function will also include any plus
or minus sign appearing in the beginning of the string, as well as any decimals that are present.

Example:
void  main()
{
string  my_string="-35.6";
double  my_number=string_to_number(my_string);
}

string_to_upper_case

This function converts all lower case characters in a string to upper case.

string string_to_upper_case(string the_string)

Parameters:
the_string
The string that will be processed.

Return value:
Returns a string with all lower case letters converted to upper case, or the original string on failure.

Remarks:
All punctuation and upper case characters present in the source string will be kept in tact.

Example:
void  main()
{
alert("String_to_upper_case  test",  string_to_upper_case("I  will  be  converted  to  upper  case."));
}

string_trim_left

This function truncates a specified number of characters from the left hand side of a string.

string string_trim_left(string the_string, uint count)

Parameters:
the_string
The string that will be extracted from.
count
The number of characters to remove.

Return value:
Returns the string with so many characters removed from the left.

Remarks:
If position is less than 1, the string is returned in its entirety. If position is greater than the string's length, a blank string will be returned.

Example:
void  main()
{
alert("string_trim_left  test",  string_trim_left("Hello,  I  am  a  string!",  7));  //output  should  be  I  am  a  string!
}

string_trim_right

This function truncates a specified number of characters from the right hand side of a string.

string string_trim_right(string the_string, uint count)

Parameters:
the_string
The string that will be extracted from.
count
The number of characters to remove.

Return value:
Returns the string with so many characters removed from the right.

Remarks:
If position is less than 1, the string is returned in its entirety. If position is greater than the string's length, a blank string will be returned.

Example:
void  main()
{
alert("string_trim_right  test",  string_trim_right("Hello,  I  am  a  string!",  16));  //output  should  be  Hello
}

array object

The array object is used to store multiple values of the same type.

array(int number_of_elements)

Parameters:
number_of_elements
An optional parameter that specifies the number of elements to create.

Remarks:
The array object is unique to any other object, since it is declared as and behaves like a variable containing a single value. When
you wish to declare an array, you simply put the type of variable that the array will hold, followed by an empty set of square
brackets. Then when you wish to access one of its values, you put the element number in a set of square brackets (see the
examples and language tutorial for a more in-depth explanation of the workings of arrays).

Please note that if an error occurs while working with arrays, the BGT engine will terminate with a runtime error rather than
silently setting an error code. Therefore it is essential that you take great care when working with arrays.

Example:
// Declare an array and display its values.

void  main()
{
string[]  names(5);
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
for(uint  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

The array object is used to store multiple values of the same type.

array(int number_of_elements)

Parameters:
number_of_elements
An optional parameter that specifies the number of elements to create.

Remarks:
The array object is unique to any other object, since it is declared as and behaves like a variable containing a single value. When
you wish to declare an array, you simply put the type of variable that the array will hold, followed by an empty set of square
brackets. Then when you wish to access one of its values, you put the element number in a set of square brackets (see the
examples and language tutorial for a more in-depth explanation of the workings of arrays).

Please note that if an error occurs while working with arrays, the BGT engine will terminate with a runtime error rather than
silently setting an error code. Therefore it is essential that you take great care when working with arrays.

Example:
// Declare an array and display its values.

void  main()
{
string[]  names(5);
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
for(uint  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

This method finds a value in the array and returns the index.

int find(uint start_index, ? value)

Parameters:
start_index
The index to start searching from.
value
The value to find.

Return value:
The position of the value on success, and -1 on failure.

Remarks:
This method is overloaded, and can be called omitting the start_index parameter. If this is the case, start_index will be assumed
to be 0.
The value parameter must match the data type being held by the array.

Please note: To search an array of classes it is necessary for the class to overload the comparison operator.

Example:
// Declare an array and find a value, displaying its index.

void  main()
{
string[]  names(6);
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
names[5]="Edmund";
names.sort_ascending(0,  names.length());
int  index=names.find("Philip");
if(index<0)
{
alert("Uh-oh",  "Why  can't  we  find  Philip?");
}
else
{
alert("Names",  "Philip  is  positioned  at  index  "+index);
}
}

array object

This method returns the length of the array.

uint get_size()

Parameters:
None.

Return value:
The length of the array, or 0 if the array is empty or if an error occurs.

Remarks:
Please note that the value returned is the length of the array rather than the last element index in the array. Therefore when using
the get_size method as a basis for a loop it is important that you check that any variable counting through the array stops when
the counter reaches the length mark, not beyond it. This is because the index is 0 based. In other words, with an array containing
3 entries you may access elements 0, 1 and 2.

Example:
// Declare an array and display its values.

void  main()
{
string[]  names(5);
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
alert("Count",  "The  array  contains  "  +  names.length()  +  "  elements.");
for(int  counter=0;  counter<names.get_size();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

This method will insert a new value at a specified position in an array.

void insert_at(int position, ? value)

Parameters:
position
The index at which the new element should be inserted.
value
A value of the specified type in the array declaration to place at that index.

Return value:
None.

Remarks:
The method can only insert values from element 0 to the previous return value of length, that is to say one more than the upper
boundary.

Example:
string[]  names(5);

void  main()
{
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
names.insert_at(5,"Edmund");
for(uint  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

This method will insert a new value at the end of the array.

void insert_last(? value)

Parameters:
value
A value of the specified type in the array declaration to place at that index.

Return value:
None.

Remarks:
None.

Example:
string[]  names(5);

void  main()
{
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
names.insert_last("Edmund");
for(uint  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

This method is used to determine whether the given array is empty. That is to say, contains no elements.

bool is_empty()

Parameters:
None.

Return value:
True if the array is empty, false otherwise.

Remarks:
None.

Example:
// Declare an array and check if it is empty.

void  main()
{
string[]  names(5);
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
if(names.is_empty())
{
alert("Oh  dear!",  "The  array  seems  to  be  empty,  despite  attempting  to  fill  it.");
exit();
}
alert("Information",  "The  array  is  nicely  full  of  data.");
}

array object

This method returns the length of the array.

uint length()

Parameters:
None.

Return value:
The length of the array, or 0 if the array is empty or if an error occurs.

Remarks:
Please note that the value returned is the length of the array rather than the last element index in the array. Therefore when using
the length method as a basis for a loop it is important that you check that any variable counting through the array stops when the
counter reaches the length mark, not beyond it. This is because the index is 0 based. In other words, with an array containing 3
entries you may access elements 0, 1 and 2.

Example:
// Declare an array and display its values.

void  main()
{
string[]  names(5);
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
alert("Count",  "The  array  contains  "  +  names.length()  +  "  elements.");
for(int  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

This method will remove an existing value at a specified position in an array.

void remove_at(int position)

Parameters:
position
The index of the element to remove.

Return value:
None.

Remarks:
Once the value has been removed the array will be automatically resized to reflect the change.

Example:
string[]  names(6);

void  main()
{
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
names[5]="Edmund";
names.remove_at(5);
for(uint  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

This method will remove the final value in the array.

void remove_last()

Parameters:
None.

Return value:
None.

Remarks:
On the outside, this method simply does the equivalent of array.resize(array.length()-1).

Example:
string[]  names(6);

void  main()
{
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
names[5]="Edmund";
names.remove_last();
for(uint  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

This method reserves memory ready for holding items in an array object.

void reserve(uint size)

Parameters:
size
The reserve size of the array.

Return value:
None.

Remarks:
If memory is already reserved for the array, and you call it with a number greater than your previous one, it will increase the reserved
memory. If the number is less than your original size, it will have no effect.

Please note: calling the reserve method on the array only reserves the amount of memory needed internally to hold the data and is used
mainly to increase performance when using the insert methods. To change the number of indexes that can be used to access the array,
please use the resize method instead.

Example:
// Compare performance between two arrays. One does not reserve memory, the other one does.

void  main()
{
timer  counter;
int[]  list;
int[]  list2;
for(int  i=0;i<100000;i++)
list.insert_last(i);
alert("First  result",  "Time  elapsed  without  reserved  memory:  "  +  counter.elapsed  +  "  milliseconds.");
list.resize(0);
counter.restart();
list2.reserve(100000);
for(int  i=0;i<100000;i++)
list2.insert_last(i);
alert("Second  result",  "Time  elapsed  with  reserved  memory:  "  +  counter.elapsed  +  "  milliseconds.");
}

array object

This method resizes an array object.

void resize(uint new_length)

Parameters:
new_length
The new length of the array. A value of 0 means the array is reset completely.

Return value:
None.

Remarks:
If the new_length parameter is greater than the current length, the relevant number of empty values will be added to the array. If
the value is less, the array is truncated from right to left.

Example:
// Declare an array and display its values.

void  main()
{
string[]  names(5);
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
names.resize(names.length()+1);
names[5]="Edmund";
for(uint  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

This method will reverse the order of the array.

void reverse()

Parameters:
None.

Return value:
None.

Remarks:
None.

Example:
//  Declare  an  array  and  reverse  the  order,  displaying  the  results.

void  main()
{
string[]  names(6);
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
names[5]="Edmund";
names.reverse();
for(int  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

This method sorts the items in an array in ascending order.

void sort_ascending(uint start_index, uint count)

Parameters:
start_index
The index to begin sorting from.
count
The number of elements to sort.

Return value:
None.

Remarks:
This method is overloaded, and can be called with no parameters. If this is the case, the entire array will be sorted.

The start_index parameter is 0-based.

Please note: To sort an array of classes it is necessary for the class to overload the comparison operator.

Example:
// Declare an array, sort its values and display them. We will use the extended declaration to show how to manually sort the array.

void  main()
{
string[]  names(6);
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
names[5]="Edmund";
names.sort_ascending(0,  names.length());
for(int  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

array object

This method sorts the items in an array in descending order.

void sort_descending(uint start_index, uint count)

Parameters:
start_index
The index to begin sorting from.
count
The number of elements to sort.

Return value:
None.

Remarks:
This method is overloaded, and can be called with no parameters. If this is the case, the entire array will be sorted.

The start_index parameter is 0-based.

Please note: To sort an array of classes it is necessary for the class to overload the comparison operator.

Example:
// Declare an array, sort its values and display them. We will use the extended declaration to show how to manually sort the array.

void  main()
{
string[]  names(6);
names[0]="Damien";
names[1]="Philip";
names[2]="Percival";
names[3]="Byron";
names[4]="Humphrey";
names[5]="Edmund";
names.sort_descending(0,  names.length());
for(int  counter=0;  counter<names.length();  counter++)
{
alert("Names",  "Element  "+counter+"  contains  the  name  "+names[counter]+".");
}
}

calendar object

The calendar object is used to calculate and manipulate dates and times.

calendar()

Parameters:
None.

Remarks:
When created, the calendar object is automatically set to the current system date and time.

This object uses operator overloading so that dates can be compared accurately. Below is a list of the operators that can be
used.

=
<
<=
>
>=
==
!=

Example:
//  Do  some  comparisons  with  calendars.

void  main()
{
calendar  now;
calendar  alarm;
timer  time;
alarm.add_seconds(10);
while(true)
{
wait(5);
if(time.elapsed>=1000)
{
time.restart();
now.reset();
}
if(now==alarm)
{
alert("Alarm",  "This  is  an  alarm!");
exit();
}
}
}

calendar object

The calendar object is used to calculate and manipulate dates and times.

calendar()

Parameters:
None.

Remarks:
When created, the calendar object is automatically set to the current system date and time.

This object uses operator overloading so that dates can be compared accurately. Below is a list of the operators that can be
used.

=
<
<=
>
>=
==
!=

Example:
//  Do  some  comparisons  with  calendars.

void  main()
{
calendar  now;
calendar  alarm;
timer  time;
alarm.add_seconds(10);
while(true)
{
wait(5);
if(time.elapsed>=1000)
{
time.restart();
now.reset();
}
if(now==alarm)
{
alert("Alarm",  "This  is  an  alarm!");
exit();
}
}
}

calendar object

This method will add a number of days to the day property.

bool add_days(int days)

Parameters:
days
The number of days to add.

Return value:
true on success, false on failure.

Remarks:
A negative value in the days parameter will subtract days from the current day.

Any number of days can be added or subtracted. The month and year properties will adjust accordingly.

Example:
// Create a calendar and set it to October 7, 1767 at 09:00 and add a week.

void  main()
{
calendar  test;
test.set(1767, 10, 7, 9, 0, 0);
test.add_days(7);
alert("Date",  "The  date  is  "+test.weekday_name+",  "+test.month_name+"  "+test.day+",  "+test.year+",  at  "+test.hour+":"+test.minute+":"+test.second);
}

calendar object

This method will add a number of hours to the hour property.

bool add_hours(int hours)

Parameters:
hours
The number of hours to add.

Return value:
true on success, false on failure.

Remarks:
A negative value in the hours parameter will subtract hours from the current hour.

Any number of hours can be added or subtracted. The date values will adjust themselves accordingly.

Example:
// Create a calendar and set it to October 7, 1767 at 09:00 and add three hours.

void  main()
{
calendar  test;
test.set(1767, 10, 7, 9, 0, 0);
test.add_hours(3);
alert("Date",  "The  date  is  "+test.weekday_name+",  "+test.month_name+"  "+test.day+",  "+test.year+",  at  "+test.hour+":"+test.minute+":"+test.second);
}

calendar object

This method will add a number of minutes to the minute property.

bool add_minutes(int minutes)

Parameters:
minutes
The number of minutes to add.

Return value:
true on success, false on failure.

Remarks:
A negative value in the minutes parameter will subtract minutes from the current minute.

Any number of minutes can be added or subtracted. The rest of the time will adjust accordingly.

Example:
// Create a calendar and set it to October 7, 1767 at 09:00 and add 60 minutes.

void  main()
{
calendar  test;
test.set(1767, 10, 7, 9, 0, 0);
test.add_minutes(60);
alert("Date",  "The  date  is  "+test.weekday_name+",  "+test.month_name+"  "+test.day+",  "+test.year+",  at  "+test.hour+":"+test.minute+":"+test.second);
}

calendar object

This method will add a number of months to the month property.

bool add_months(int months)

Parameters:
months
The number of months to add.

Return value:
true on success, false on failure.

Remarks:
A negative value in the months parameter will subtract months from the current month.

There are several cases where values can change. If the date property is not valid for the new month, the date is set to the last day of the new month. If the month itself is not valid for the current
year, then the year and month will be adjusted accordingly.

Example:
// Create a calendar and set it to October 7, 1767 at 09:00 and add four months.

void  main()
{
calendar  test;
test.set(1767, 10, 7, 9, 0, 0);
test.add_months(4);
alert("Date",  "The  date  is  "+test.weekday_name+",  "+test.month_name+"  "+test.day+",  "+test.year+",  at  "+test.hour+":"+test.minute+":"+test.second);
}

calendar object

This method will add a number of seconds to the second property.

bool add_seconds(int seconds)

Parameters:
seconds
The number of seconds to add.

Return value:
true on success, false on failure.

Remarks:
A negative value in the seconds parameter will subtract seconds from the current second.

Any number of seconds can be added or subtracted. The rest of the time will adjust accordingly.

Example:
// Create a calendar and set it to October 7, 1767 at 09:00 and add 86400 seconds.

void  main()
{
calendar  test;
test.set(1767, 10, 7, 9, 0, 0);
test.add_seconds(86400);
alert("Date",  "The  date  is  "+test.weekday_name+",  "+test.month_name+"  "+test.day+",  "+test.year+",  at  "+test.hour+":"+test.minute+":"+test.second);
}

calendar object

This method will add a number of years to the year property.

bool add_years(int years)

Parameters:
years
The number of years to add.

Return value:
true on success, false on failure.

Remarks:
A negative value in the years parameter will subtract years from the current year.

In most cases, the month and date will not change. However, if the date is February 29, and the adjusted year property is not a leap year, the calendar will set the date to March 1.

Example:
// Create a calendar and set it to October 7, 1767 at 09:00 and add a year.

void  main()
{
calendar  test;
test.set(1767, 10, 7, 9, 0, 0);
test.add_years(1);
alert("Date",  "The  date  is  "+test.weekday_name+",  "+test.month_name+"  "+test.day+",  "+test.year+",  at  "+test.hour+":"+test.minute+":"+test.second);
}

calendar object

This method returns the number of days between two dates.

double diff_days(calendar@ object)

Parameters:
object
The handle to another valid calendar object.

Return value:
The difference between the current calendar's date and the date of the passed calendar object handle in days.

Remarks:
Unlike diff_years, this method returns an integer representation of the difference.

Example:
// Find out the number of days between October 7, 1767 and now.

void  main()
{
calendar  test;
calendar  now;
test.set(1767, 10, 7, 9, 0, 0);
double  diff=test.diff_days(now);
alert("diff_days",  "There  are  "+diff+"  days  between  October  7,  1767  and  now.");
}

calendar object

This method returns the number of hours between two dates.

double diff_hours(calendar@ object)

Parameters:
object
The handle to another valid calendar object.

Return value:
The difference between the current calendar's date and the date of the passed calendar object handle in hours.

Remarks:
Unlike diff_years, this method returns an integer representation of the difference.

Example:
// Find out the number of hours between October 7, 1767 and now.

void  main()
{
calendar  test;
calendar  now;
test.set(1767, 10, 7, 9, 0, 0);
double  diff=test.diff_hours(now);
alert("diff_hours",  "There  are  "+diff+"  hours  between  October  7,  1767  and  now.");
}

calendar object

This method returns the number of minutes between two dates.

double diff_minutes(calendar@ object)

Parameters:
object
The handle to another valid calendar object.

Return value:
The difference between the current calendar's date and the date of the passed calendar object handle in minutes.

Remarks:
Unlike diff_years, this method returns an integer representation of the difference.

Example:
// Find out the number of minutes between October 7, 1767 and now.

void  main()
{
calendar  test;
calendar  now;
test.set(1767, 10, 7, 9, 0, 0);
double  diff=test.diff_minutes(now);
alert("diff_minutes",  "There  are  "+diff+"  minutes  between  October  7,  1767  and  now.");
}

calendar object

This method returns the number of months between two dates.

double diff_months(calendar@ object)

Parameters:
object
The handle to another valid calendar object.

Return value:
The difference between the current calendar's date and the date of the passed calendar object handle in months.

Remarks:
Unlike diff_years, this method returns an integer representation of the difference.

Example:
// Find out the number of months between October 7, 1767 and now.

void  main()
{
calendar  test;
calendar  now;
test.set(1767, 10, 7, 9, 0, 0);
double  diff=test.diff_months(now);
alert("diff_months",  "There  are  "+diff+"  months  between  October  7,  1767  and  now.");
}

calendar object

This method returns the number of seconds between two dates.

double diff_seconds(calendar@ object)

Parameters:
object
The handle to another valid calendar object.

Return value:
The difference between the current calendar's date and the date of the passed calendar object handle in seconds.

Remarks:
Unlike diff_years, this method returns an integer representation of the difference.

Example:
// Find out the number of seconds between October 7, 1767 and now.

void  main()
{
calendar  test;
calendar  now;
test.set(1767, 10, 7, 9, 0, 0);
double  diff=test.diff_seconds(now);
alert("diff_seconds",  "There  are  "+diff+"  seconds  between  October  7,  1767  and  now.");
}

calendar object

This method returns the number of years between two dates.

double diff_years(calendar@ object)

Parameters:
object
The handle to another valid calendar object.

Return value:
The difference between the current calendar's date and the date of the passed calendar object handle in years.

Remarks:
The representation of the year difference includes decimals.

Example:
// Find out the number of years between October 7, 1767 and now.

void  main()
{
calendar  test;
calendar  now;
test.set(1767, 10, 7, 9, 0, 0);
double  diff=test.diff_years(now);
alert("diff_years",  "There  are  "+diff+"  years  between  October  7,  1767  and  now.");
}

calendar object

This method will reset the calendar object to the current system time.

bool reset()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
None.

Example:
// Create a calendar and set it to October 7, 1767 at 09:00 and reset it.

void  main()
{
calendar  test;
test.set(1767, 10, 7, 9, 0, 0);
test.reset();
alert("Date",  "The  date  is  "+test.weekday_name+",  "+test.month_name+"  "+test.day+",  "+test.year+",  at  "+test.hour+":"+test.minute+":"+test.second);
}

calendar object

This method will change the current date in the calendar.

bool set(int year, int month, int day, int hour, int minute, int second)

Parameters:
year
The year, between 0 and 9999.
month
The month between 1 and 12.
day
The date, between 1 and 31.
hour
The hour, between 0 and 23.
minute
The minute, between 0 and 59.
second
The second, between 0 and 59.

Return value:
true on success, false on failure.

Remarks:
This method does not change the system time, but rather it changes the date properties of the calendar object itself.

Example:
// Create a calendar and set it to October 7, 1767 at 09:00.

void  main()
{
calendar  test;
test.set(1767, 10, 7, 9, 0, 0);
alert("Date",  "The  date  is  "+test.weekday_name+",  "+test.month_name+"  "+test.day+",  "+test.year+",  at  "+test.hour+":"+test.minute+":"+test.second);
}

calendar object

bool leap_year
This property is set to true if the current year is a leap year, false otherwise.

int year
This shows the year currently held by the calendar object.

int month
This shows the month currently held by the calendar object.

string month_name
This converts the currently set month into its English name.

int day
This shows the date currently held by the calendar object.

int weekday
This shows the day of the week currently held by the calendar object, where 1=Monday and 7=Sunday.

string weekday_name
This converts the currently set weekday into its English name.

int hour
This shows the hour currently held by the calendar object.

int minute
This shows the minute currently held by the calendar object.

int second
This shows the second currently held by the calendar object.

bool valid
This is set to true if the current date in the calendar object is valid, false otherwise.

combination object

The combination object contains algorithms to enumerate arbitrarily large combinations of elements in a given set.

combination()

Parameters:
None.

Remarks:
Enumerating combinations of elements is useful in many different scenarios. An example is when you have a set of sports teams, all of which must play against every other team at least once but no more than
that. Another example which requires a different type of combination is the traditional game of rock, paper and sizzors. There we want to find each combination of two that the three items can form, in every
possible order including the same item appearing twice.

The combination object facilitates exactly these types of enumerations. You start by calling one of the generate_x methods depending on which algorithm you want, and then you repeatedly call the next
method to retrieve each combination in an array. You continue doing this until the next method returns false. At this point you will have enumerated all possible combinations that the algorithm searches for,
and your array will be empty.

When you call one of the generate_x methods you need to specify the number of items that you have in your set, as well as how many items you want in each combination. Depending on the behavior of the
algorithm in question, there will be different restrictions for these values. When the next method fills your array with a new combination, it will use the numbers in your set to denote the items in that
combination. These values will range from 0 to the number of items in your set minus 1.

Note that many of these algorithms have the potential to quickly generate millions or billions of combinations depending on the size of your set and the number of items in the combinations. The examples
purposefully use small values, so that they can be printed in alert boxes easily. Be very careful when you start working with larger sets and/or combinations, therefore, so that you don't attempt to display a
million message boxes.

Example:
// We have 4 sports teams and we want each of them to play against every other team once, but no more than that.

void  main()
{
int  combination_size=2;
// This is the number of items that we want in each combination.

int  items=4;
//  This  is  the  number  of  items  we  have  all  in  total,  which  is  to  say  our  sports  teams.

combination  process;

if(process.generate_unique_combinations(items,  combination_size)==false)
{
alert("Error",  get_last_error_text());
exit();
}

int[]  list;
// This is the array which will be filled with each combination.

int  count=0;
// This simply keeps track of how many combinations we've generated so far. It is not really needed.

while(process.next(list))
{
count+=1;

// At this point, we have a new combination. We print it out in an alert box as a comma separated string.
string  data;
for(int  i=0;i<list.length();i++)
{
if(i>0)
data+=",  ";

data+=list[i];
}
alert("Combination  "  +  count,  data);
}

// We're done, so let's display how many combinations we found before we quit.
alert("Result", count + " unique combinations were generated from a set of " + items + " items, with " + combination_size + " items in each combination.");
}

combination object

The combination object contains algorithms to enumerate arbitrarily large combinations of elements in a given set.

combination()

Parameters:
None.

Remarks:
Enumerating combinations of elements is useful in many different scenarios. An example is when you have a set of sports teams, all of which must play against every other team at least once but no more than
that. Another example which requires a different type of combination is the traditional game of rock, paper and sizzors. There we want to find each combination of two that the three items can form, in every
possible order including the same item appearing twice.

The combination object facilitates exactly these types of enumerations. You start by calling one of the generate_x methods depending on which algorithm you want, and then you repeatedly call the next
method to retrieve each combination in an array. You continue doing this until the next method returns false. At this point you will have enumerated all possible combinations that the algorithm searches for,
and your array will be empty.

When you call one of the generate_x methods you need to specify the number of items that you have in your set, as well as how many items you want in each combination. Depending on the behavior of the
algorithm in question, there will be different restrictions for these values. When the next method fills your array with a new combination, it will use the numbers in your set to denote the items in that
combination. These values will range from 0 to the number of items in your set minus 1.

Note that many of these algorithms have the potential to quickly generate millions or billions of combinations depending on the size of your set and the number of items in the combinations. The examples
purposefully use small values, so that they can be printed in alert boxes easily. Be very careful when you start working with larger sets and/or combinations, therefore, so that you don't attempt to display a
million message boxes.

Example:
// We have 4 sports teams and we want each of them to play against every other team once, but no more than that.

void  main()
{
int  combination_size=2;
// This is the number of items that we want in each combination.

int  items=4;
//  This  is  the  number  of  items  we  have  all  in  total,  which  is  to  say  our  sports  teams.

combination  process;

if(process.generate_unique_combinations(items,  combination_size)==false)
{
alert("Error",  get_last_error_text());
exit();
}

int[]  list;
// This is the array which will be filled with each combination.

int  count=0;
// This simply keeps track of how many combinations we've generated so far. It is not really needed.

while(process.next(list))
{
count+=1;

// At this point, we have a new combination. We print it out in an alert box as a comma separated string.
string  data;
for(int  i=0;i<list.length();i++)
{
if(i>0)
data+=",  ";

data+=list[i];
}
alert("Combination  "  +  count,  data);
}

// We're done, so let's display how many combinations we found before we quit.
alert("Result", count + " unique combinations were generated from a set of " + items + " items, with " + combination_size + " items in each combination.");
}

combination object

This algorithm generates all combinations of a given size from a set, including repetitions in all possible orders.

bool generate_all_combinations(int items, int combination_size)

Parameters:
items
The number of items in the set. This must be at least 2.
combination_size
The number of items in each combination. Since we include repetitions, this must be at least 2 but can be greater than items.

Return value:
true on success, false on failure.

Remarks:
When enumerating combinations with two items in each from a set of three items, the nine possible combinations that this algorithm generates are:

[ 0, 0 ]
[ 0, 1 ]
[ 0, 2 ]
[ 1, 0 ]
[ 1, 1 ]
[ 1, 2 ]
[ 2, 0 ]
[ 2, 1 ]
[ 2, 2 ]

Example:
// We have a basket and an unlimited quantity of three types of fruit. The basket may only contain two fruits at any given time.
//  Enumerate  all  the  combinations  of  fruit  that  the  basket  could  possibly  contain.
// Note: This same algorithm can be used for other things such as getting all the possible outcomes of rock, paper and sizzors.

void  main()
{
int  combination_size=2;
// This is the number of items that we want in each combination.

int  items=3;
// This is the number of items we have all in total, which is to say our different types of fruit.

combination  process;

if(process.generate_all_combinations(items,  combination_size)==false)
{
alert("Error",  get_last_error_text());
exit();
}

int[]  list;
// This is the array which will be filled with each combination.

int  count=0;
// This simply keeps track of how many combinations we've generated so far. It is not really needed.

while(process.next(list))
{
count+=1;

// At this point, we have a new combination. We print it out in an alert box as a comma separated string.
string  data;

for(int  i=0;i<list.length();i++)
{
if(i>0)
data+=",  ";
data+=list[i];
}
alert("Combination  "  +  count,  data);
}

// We're done, so let's display how many combinations we found before we quit.
alert("Result", count + " combinations were generated from a set of " + items + " items, with " + combination_size + " items in each combination.");
}

combination object

This algorithm generates all unique combinations of a given size from a set.

bool generate_unique_combinations(int items, int combination_size)

Parameters:
items
The number of items in the set. This must be at least 3.
combination_size
The number of items in each combination. Since the generated combinations must be unique, this value must be greater than 1 and less than items.

Return value:
true on success, false on failure.

Remarks:
When enumerating combinations with two items in each from a set of four items, the six possible combinations that this algorithm generates are:

[ 0, 1 ]
[ 0, 2 ]
[ 0, 3 ]
[ 1, 2 ]
[ 1, 3 ]
[ 2, 3 ]

Example:
// We have 4 sports teams and we want each of them to play against every other team once, but no more than that.

void  main()
{
int  combination_size=2;
// This is the number of items that we want in each combination.

int  items=4;
//  This  is  the  number  of  items  we  have  all  in  total,  which  is  to  say  our  sports  teams.

combination  process;

if(process.generate_unique_combinations(items,  combination_size)==false)
{
alert("Error",  get_last_error_text());
exit();
}

int[]  list;
// This is the array which will be filled with each combination.

int  count=0;
// This simply keeps track of how many combinations we've generated so far. It is not really needed.

while(process.next(list))
{
count+=1;

// At this point, we have a new combination. We print it out in an alert box as a comma separated string.
string  data;
for(int  i=0;i<list.length();i++)
{
if(i>0)
data+=",  ";
data+=list[i];
}

alert("Combination  "  +  count,  data);
}

// We're done, so let's display how many combinations we found before we quit.
alert("Result", count + " unique combinations were generated from a set of " + items + " items, with " + combination_size + " items in each combination.");
}

combination object

This method fills an array with the next combination in the sequence generated by the currently used algorithm.

bool next(int[]@ list)

Parameters:
list
An array of ints which will be automatically resized if necessary, and then filled with the next combination in the sequence.

Return value:
true if the array was successfully filled with the next combination, false if there are no more combinations or if an error occurred.

Remarks:
When calling this method, you should always check the return value. if true is returned, it means that your array is now populated with the next combination in the sequence as dictated by the
algorithm that you are using. If false is returned and no error is set, it means that all combinations were enumerated successfully.

The reason this method takes the array as an argument rather than returning it is to avoid the necessity of making a new array instance each time. Since this method has the potential of being called
many times, it is important to avoid as much unnecessary processing as possible (such as making a new array which requires memory allocations).

Example:
// We have a basket and an unlimited quantity of three types of fruit. The basket may only contain two fruits at any given time.
//  Enumerate  all  the  combinations  of  fruit  that  the  basket  could  possibly  contain.
// Note: This same algorithm can be used for other things such as getting all the possible outcomes of rock, paper and sizzors.

void  main()
{
int  combination_size=2;
// This is the number of items that we want in each combination.

int  items=3;
// This is the number of items we have all in total, which is to say our different types of fruit.

combination  process;

if(process.generate_all_combinations(items,  combination_size)==false)
{
alert("Error",  get_last_error_text());
exit();
}

int[]  list;
// This is the array which will be filled with each combination.

int  count=0;
// This simply keeps track of how many combinations we've generated so far. It is not really needed.

while(process.next(list))
{
count+=1;

// At this point, we have a new combination. We print it out in an alert box as a comma separated string.
string  data;
for(int  i=0;i<list.length();i++)
{
if(i>0)
data+=",  ";
data+=list[i];
}
alert("Combination  "  +  count,  data);
}

// We're done, so let's display how many combinations we found before we quit.

alert("Result", count + " combinations were generated from a set of " + items + " items, with " + combination_size + " items in each combination.");
}

combination object

This method resets the state of the given combination algorithm (if any).

void reset()

Parameters:
None.

Return value:
None.

Remarks:
After calling this method, you must call one of the generate_x methods before you attempt to call the next method again.

Example:
None.

combination object

bool active
This will be true while an algorithm is in the process of generating combinations. This property cannot be modified from the
script.

dictionary object

The dictionary object is used to store pairs of keys and values, the key being a string with which one can look up a value of any
type.

dictionary()

Parameters:
None.

Remarks:
The dictionary object comes in handy when you need to store an arbitrary number of variables with different types. When setting
a new value you must use a unique string key with which you can then refer to the value again. You are able to delete a value at
any given time, without affecting any of the others.

In more technical terms, the dictionary is what is commonly known as a hash map. When a new key/data pare is inserted, a hash
is generated from the key which makes later lookup operations much faster than if a linear search of all the keys had to be
performed.

Example:
dictionary  game_board;  //  Declare  our  dictionary.

void  main()
{
int  value;  //  Declare  a  variable  for  retrieving  values  from  a  dictionary.

//  Set  our  values.
game_board.set("1", 2); // Sets the value 2 to a key called 1.
game_board.set("2", 4); // Sets the value 4 to a key called 2.
game_board.get("1",  value);  //  Retrieve  value  for  1.
alert("Dictionary  example",  "current  value="+value);
game_board.get("2",  value);  //  Retrieve  value  for  2.
alert("Dictionary  example",  "current  value="+value);
game_board.delete_all();  //  Delete  all  the  values.
}

dictionary object

The dictionary object is used to store pairs of keys and values, the key being a string with which one can look up a value of any
type.

dictionary()

Parameters:
None.

Remarks:
The dictionary object comes in handy when you need to store an arbitrary number of variables with different types. When setting
a new value you must use a unique string key with which you can then refer to the value again. You are able to delete a value at
any given time, without affecting any of the others.

In more technical terms, the dictionary is what is commonly known as a hash map. When a new key/data pare is inserted, a hash
is generated from the key which makes later lookup operations much faster than if a linear search of all the keys had to be
performed.

Example:
dictionary  game_board;  //  Declare  our  dictionary.

void  main()
{
int  value;  //  Declare  a  variable  for  retrieving  values  from  a  dictionary.

//  Set  our  values.
game_board.set("1", 2); // Sets the value 2 to a key called 1.
game_board.set("2", 4); // Sets the value 4 to a key called 2.
game_board.get("1",  value);  //  Retrieve  value  for  1.
alert("Dictionary  example",  "current  value="+value);
game_board.get("2",  value);  //  Retrieve  value  for  2.
alert("Dictionary  example",  "current  value="+value);
game_board.delete_all();  //  Delete  all  the  values.
}

dictionary object

This method will attempt to delete a particular key and value pair.

bool delete(string key)

Parameters:
key
The key to look for.

Return value:
true on success, false on failure.

Remarks:
None.

Example:
dictionary  game_board;  //  Declare  our  dictionary.

void  main()
{
//  Set  our  values.
game_board.set("1",  2);
game_board.set("2",  4);
game_board.delete("1");
}

dictionary object

This method will delete all key and value pairs currently present in the dictionary.

bool delete_all()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
None.

Example:
// Set some values, then delete them all.

dictionary  game_board;  //  Declare  our  dictionary.

void  main()
{
//  Set  our  values.
game_board.set("1", 2); // Sets the value 2 to a key called 1.
game_board.set("2", 4); // Sets the value 4 to a key called 2.
game_board.delete_all();  //  Delete  all  the  values.
}

dictionary object

This method will check to see if a particular key exists in the dictionary.

bool exists(string key)

Parameters:
key
The key to look for.

Return value:
true if the key was found, false if it was not or if an error occured.

Remarks:
None.

Example:
dictionary  game_board;  //  Declare  our  dictionary.

void  main()
{
int  value;  //  Declare  a  variable  for  retrieving  values  from  a  dictionary.
//  Set  our  values.
game_board.set("1", 2); // Sets the value 2 to a key called 1.
game_board.set("2", 4); // Sets the value 4 to a key called 2.
if(game_board.exists("1"))
{
game_board.get("1",  value);  //  Retrieve  value  for  1.
alert("Dictionary  example",  "Key  1  contains  value  "+value);
}
else
{
alert("Error",  "Key  1  does  not  exist  in  the  dictionary.");
}
game_board.delete_all();  //  Delete  all  the  values.
}

dictionary object

This method will retrieve a value based on the specified key.

bool get(string key, ? &out value)

Parameters:
key
The key to look for.
value
This should be the name of a variable of any type which will receive the value if it is found (see remarks).

Return value:
true on success, false on failure.

Remarks:
The value parameter is declared with the &out keyword since it is a parameter that the function will write to, not read from (see
the section on functions in the language tutorial for more details). Make certain that the value that you stored in the dictionary has
the same type as the variable into which you tell the function to write the value, as a mismatch here will cause the function to fail.

Example:
dictionary  game_board;  //  Declare  our  dictionary.

void  main()
{
int  value;  //  Declare  a  variable  for  retrieving  values  from  the  dictionary.
//  Set  our  values.
game_board.set("1", 2); // Sets the value 2 to a key called 1.
game_board.set("2", 4); // Sets the value 4 to a key called 2.
game_board.get("1",  value);  //  Retrieve  value  for  1.
alert("Dictionary  example",  "current  value="+value);
game_board.get("2",  value);  //  Retrieve  value  for  2.
alert("Dictionary  example",  "current  value="+value);
game_board.delete_all();  //  Delete  all  the  values.
}

dictionary object

This method will retrieve a list of all the keys that are present in the dictionary.

string[] get_keys()

Parameters:
None.

Return value:
An array with all the keys on success, or an empty array on failure.

Remarks:
Internally the BGT engine uses what is known as a hash map for the dictionary, which means that the time that it takes to look up
values is greatly decreased at the cost of losing ordering. This means that the keys returned by this method may appear in any
order, which is likely to be very different from the order in which you actually added them.

Example:
dictionary  game_board;  //  Declare  our  dictionary.

void  main()
{
// Add a bunch of keys to the dictionary.
for(uint  i=0;i<20;i++)
{
game_board.set("key_"  +  (i+1),  0);
}

//  Assemble  a  string  with  a  list  of  all  the  keys  and  display  it  in  an  alert  box.
string  output;
string[]  keys=game_board.get_keys();
for(uint  i=0;i<keys.length();i++)
{
if(output=="")
output+=keys[i];
else
output+="\n"  +  keys[i];
}
alert("Keys",  output);
}

dictionary object

This method returns the number of elements in the dictionary.

uint get_size()

Parameters:
None.

Return value:
The number of elements present in the dictionary, or 0 if the dictionary is empty or if an error occurs.

Remarks:
This method returns the number of key/value pairs in the dictionary. Therefore if the dictionary had a value named "value" linked
to a key named "key", the size would return 1.

Example:
// Create a dictionary and display its size.

void  main()
{
int  value=-1;
dictionary  names;
names.set("Damien",  0);
names.set("Philip",  1);
names.set("Percival",  2);
names.set("Byron",  3);
names.set("Humphrey",  4);
alert("Count",  "The  dictionary  contains  "  +  names.get_size()  +  "  entries.");
string[]  keys=names.get_keys();
for(int  counter=0;  counter<:keys.get_size();  counter++)
{
names.get(keys[counter],  value);
alert("Names",  keys[counter]+"  contains  the  value  "+value+".");
}
}

dictionary object

This method is used to determine whether the dictionary is empty. That is to say, contains no entries.

bool is_empty()

Parameters:
None.

Return value:
True if the dictionary is empty, false otherwise.

Remarks:
None.

Example:
// Declare a dictionary and check if it is empty.

void  main()
{
dictionary  example;
example.set("Name",  "Damien");
if(example.is_empty())
{
alert("Oh  dear!",  "The  dictionary  seems  to  be  empty,  despite  attempting  to  feed  it  with  data.");
exit();
}
alert("Information",  "The  dictionary  is  nicely  full  of  data.");
}

dictionary object

This method will set a value refered to by the key.

void set(string key, ? value)

Parameters:
key
The key to use.
value
A variable of any type who's value you wish to associate with the specified key.

Return value:
None.

Remarks:
If a particular key already exists in the dictionary, its value will be automatically overwritten.

Example:
dictionary  game_board;  //  Declare  our  dictionary.

void  main()
{
int  value;  //  Declare  a  variable  for  retrieving  values  from  a  dictionary.
//  Set  our  values.
game_board.set("1", 2); // Sets the value 2 to a key called 1.
game_board.set("2", 4); // Sets the value 4 to a key called 2.
game_board.get("1",  value);  //  Retrieve  value  for  1.
alert("Dictionary  example",  "current  value="+value);
game_board.get("2",  value);  //  Retrieve  value  for  2.
alert("Dictionary  example",  "current  value="+value);
game_board.delete_all();  //  Delete  all  the  values.
}

file object

The file object is used to read and write files stored on the hard disk.

file()

Parameters:
None.

Remarks:
The file object is initially blank, until a file is associated with it. To associate a file you must call the "open" method specifying a
name or a path for the file, and the mode in which the file is opened. You may call the "open" method again at any time if you
wish to open a new file, at which point the old file (if any) will be cleared. This, in short, allows you to reuse the same file object
to control multiple files.

Example:
// Write some text into a file.

void  main()
{
file  test;
test.open("c:\\test.txt",  "w");
test.write("I  am  a  text  file!");
test.close();
}

file object

The file object is used to read and write files stored on the hard disk.

file()

Parameters:
None.

Remarks:
The file object is initially blank, until a file is associated with it. To associate a file you must call the "open" method specifying a
name or a path for the file, and the mode in which the file is opened. You may call the "open" method again at any time if you
wish to open a new file, at which point the old file (if any) will be cleared. This, in short, allows you to reuse the same file object
to control multiple files.

Example:
// Write some text into a file.

void  main()
{
file  test;
test.open("c:\\test.txt",  "w");
test.write("I  am  a  text  file!");
test.close();
}

file object

This method will close any file associated with the object.

bool close()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
Closing a file is useful when you want to release the current file that is associated with the object in order to save resources, while
still keeping the object alive for future use. This method is executed automatically when a file object is destroyed, so you need
not worry about calling it to clean up after yourself at the end of your script for example if you do not wish to. This method is also
executed if you call the open method on a file object that already has a file associated with it.

Example:
// Open a file, write some text and close it.

void  main()
{
file  test;
test.open("c:\\test.txt",  "w");
test.write("I  am  a  text  file!");
test.close();
}

file object

This method will open a file for reading or writing.

bool open(string filename, string open_mode)

Parameters:
filename
The file to open. This can be either an absolute path, a relative path or the file name on its own.
open_mode
The mode in which the file should be opened (see remarks).

Return value:
true on success, false on failure.

Remarks:
When specifying a filename, remember that both \ and / can be used to specify paths.

The following is a list of valid open modes.

"a"=append
"r"=read
"w"=write

If the file does not exist and is opened in write or append mode, it will be created as long as the directory that should contain the
file exists. If the file is being opened in read mode then both directory and file must exist for the operation to succeed.

To open a file in binary mode, simply append b to the open mode string. For example, "wb" will open a file to be written in
binary mode, while "rb" opens a file to be read in binary mode.

Example:
// Write some text into a file.

void  main()
{
file  test;
test.open("c:\\test.txt",  "w");
test.write("I  am  a  text  file!");
test.close();
}

file object

This method will read a specified number of characters or bytes in a file.

string read(double count)

Parameters:
count
An optional number of characters/bytes to read. If this parameter is not passed or if it is set to 0, the remainder of the file will be
read from the cursor.

Return value:
A string containing the retrieved data on success, or an empty string on failure.

Remarks:
If the file is opened in text mode, the number given in the count parameter specifies the number of characters that will be read. If,
on the other hand, the file is opened as binary, then this parameter specifies the number of bytes instead.

Example:
// Read a file and display its content in an alert box.

void  main()
{
file  test;
test.open("test.txt",  "r");
alert("test.txt  contents",  test.read());
}

file object

This method will read a file until a specified string is found.

string read_until(string text, bool require_full)

Parameters:
text
The text to search for in the file.
require_full
A boolean value specifying whether the full text should be found before the file stops reading, or any character in the sequence.

Return value:
A string containing the retrieved data on success, the entire text if the file was successfully read but the string not found, or an
empty string if data is unretrievable.

Remarks:
The data returned will include the search string. If the search string cannot be found, the entire text, if any, will be returned by this
function.

Example:
// Read a file and display the first line of its content in an alert box.

void  main()
{
file  test;
test.open("test.txt",  "r");
alert("test.txt  contents",  test.read_until("\r\n",true));
}

file object

This method will move the file cursor to a new position.

bool seek(double new_position)

Parameters:
new_position
The position to seek to. 0 indicates the beginning of the file.

Return value:
true on success, false on failure.

Remarks:
If the file is opened in text mode, the number given in the new_position parameter specifies the position in characters. If, on the
other hand, the file is opened as binary, then the number specifies the position in bytes instead.

If the file is opened in write or append mode and you seek to a position prior to the end of the file and write data, the data
previously stored at that position will be overwritten.

Example:
// Read character 5 in a file.

void  main()
{
file  test;
test.open("test.txt",  "r");
test.seek(4); // We use 4 because the cursor moves one when we read or write.
alert("character  5",  "character5="+test.read(1));
}

file object

This method will write a string or text or binary data into a file.

double write(string data)

Parameters:
data
The data to be written.

Return value:
A double containing the number of characters or bytes written.

Remarks:
If the file is opened in text mode, the return value shows the number of characters written. If, on the other hand, the file is opened
as binary, then the return value shows the number of bytes written instead.

Example:
// Write to a file.

void  main()
{
file  test;
test.open("test.txt",  "w");
test.write("I  am  a  text  file!");
test.close();
}

file object

bool reached_end
This is set to true if the end of the file is reached, and is false if not at the end, or if there is no file associated with the object. This
property cannot be modified from the script.

double position
This shows the current position of the cursor. If opened in binary mode, this is given in bytes. If opened in text mode, it is the
number of characters. If no file is associated with the object or if the position cannot be retrieved, this is set to -1. This property
cannot be modified from the script.

double size
This is the size of the file. If opened in binary mode, this is given in bytes. If opened in text mode, it is the number of characters.
This is set to -1 if the size cannot be retrieved or if the file object is empty. This property cannot be modified from the script.

bool active
This is set to true when a file is associated with the object, and to false otherwise. This property cannot be modified from the
script.

http object

The http object is used to perform advanced http functions asynchronously.

http()

Parameters:
None.

Remarks:
The http object provides the same functionality as the functions url_get and url_post, as well as some further methods and properties that allow for more
advanced functionality. The biggest difference is that it performs in the background while the rest of your script can work on other things.

Example:
// Download the BGT installer to a file in the background.

binary=download.get("http://www.blastbay.com/bgt_english_installer.exe");

void  main()
{
http  download;
file  bgt;
string 
if(get_last_error()<0)
{
alert("Error",  "There  was  an  error  downloading  the  file.\nError  description:  "  +  binary  +  "\nPlease  try  again  later.");
exit();
}
bgt.open("bgt_english_installer.exe","wb");
if(bgt.active==false)
{
alert("Error",  "The  output  file  could  not  be  opened.");
exit();
}
while(download.progress)
{
binary+=download.request();
if(binary.length()>=8192)
{
bgt.write(binary);
binary="";
}
wait(5);
}
if(binary!="")
{
bgt.write(binary);
}
bgt.close();
}

http object

The http object is used to perform advanced http functions asynchronously.

http()

Parameters:
None.

Remarks:
The http object provides the same functionality as the functions url_get and url_post, as well as some further methods and properties that allow for more
advanced functionality. The biggest difference is that it performs in the background while the rest of your script can work on other things.

Example:
// Download the BGT installer to a file in the background.

binary=download.get("http://www.blastbay.com/bgt_english_installer.exe");

void  main()
{
http  download;
file  bgt;
string 
if(get_last_error()<0)
{
alert("Error",  "There  was  an  error  downloading  the  file.\nError  description:  "  +  binary  +  "\nPlease  try  again  later.");
exit();
}
bgt.open("bgt_english_installer.exe","wb");
if(bgt.active==false)
{
alert("Error",  "The  output  file  could  not  be  opened.");
exit();
}
while(download.progress)
{
binary+=download.request();
if(binary.length()>=8192)
{
bgt.write(binary);
binary="";
}
wait(5);
}
if(binary!="")
{
bgt.write(binary);
}
bgt.close();
}

http object

This method initiates the retrieval of a URL from the Internet.

string get(string url)

Parameters:
url
The URL of the file on the Internet that should be retrieved.

Return value:
A blank string on success, or an error description or blank string on failure (see remarks).

Remarks:
This method will send what is known as a GET request. A GET request is used in most cases when a normal URL is being
visited. The URL given may contain so called GET parameters, which are used to pass small amounts of data to the server.

The URL may not contain any special characters such as spaces or similar. If you wish to include such characters, always
encode the URL by calling the url_encode function before attempting to retrieve it from the Internet.

If the http server to which the URL points uses a port other than 80, you will need to include the port after the host name
proceeded by a colon. The path on the server (if any) should then follow, proceeded by a slash. Such a URL might look like
http://myserver.com:8080/myfile.html.

If multiple requests are made with URL's pointing to the same server, the http object will attempt to keep the connection alive for
a short while which results in a dramatic speed increase for subsequent requests.

If the network subsystem fails to initialize, the get_last_error function will return -27 and the returned string will be blank.

If an Internet related error occurs, the get_last_error function will return -29 and the returned string will contain a textual
description of the error. This means that you should always call get_last_error after initiating a request.

This method can be used to retrieve both binary and textual data.

The http object is asynchronous, which means that it does not wait for any data before returning. To retrieve data from the
server, you must continuously call the request method until the progress boolean property reports false.

Example:
//  Retrieve  the  contents  of  the  index  page  at  blastbay.com.

void  main()
{
http  download;
string  body=download.get("http://www.blastbay.com/");
if(get_last_error()!=0)
{
alert("Error",  "An  error  occured.\r\nDescription:  "  +  body  +  "");
}
else
{
while(download.progress)
{
body+=download.request();
wait(5);
}
clipboard_copy_text(body);
alert("Success",  "The  data  was  retrieved  and  has  been  copied  to  your  clipboard.");
}
}

http object

This method initiates the retrieval of only the http headers from a URL on the Internet.

string get_headers(string url)

Parameters:
url
The URL of the file on the Internet for which the headers should be retrieved.

Return value:
A blank string on success, or an error description or blank string on failure (see remarks).

Remarks:
This method will send what is known as a HEAD request. A HEAD request is identical to a GET request, with the only
difference being that it merely retrieves the http headers from the server that would have been returned along with the content if a
GET request with the same URL had been sent. This is useful when you wish to retrieve the size of a given file without
downloading it, for instance.

The URL given may contain so called GET parameters, which are used to pass small amounts of data to the server.

The URL may not contain any special characters such as spaces or similar. If you wish to include such characters, always
encode the URL by calling the url_encode function before attempting to retrieve it from the Internet.

If the http server to which the URL points uses a port other than 80, you will need to include the port after the host name
proceeded by a colon. The path on the server (if any) should then follow, proceeded by a slash. Such a URL might look like
http://myserver.com:8080/myfile.html.

If multiple requests are made with URL's pointing to the same server, the http object will attempt to keep the connection alive for
a short while which results in a dramatic speed increase for subsequent requests.

If the network subsystem fails to initialize, the get_last_error function will return -27 and the returned string will be blank.

If an Internet related error occurs, the get_last_error function will return -29 and the returned string will contain a textual
description of the error. This means that you should always call get_last_error after initiating a request.

The http object is asynchronous, which means that it does not wait for any data before returning. To retrieve data from the
server, you must continuously call the request method until the progress boolean property reports false. At this time, you may
access the headers property to retrieve the result.

Example:
//  Retrieve  the  size  of  the  BGT  installer  from  blastbay.com,  without  downloading  the  file.

result=download.get_headers("http://www.blastbay.com/bgt_english_installer.exe");

void  main()
{
http  download;
string 
if(get_last_error()!=0)
{
alert("Error",  "An  error  occured.\r\nDescription:  "  +  result  +  "");
}
else
{
while(download.progress)
{
download.request();
wait(5);
}

string[]  headers=string_split(download.headers,  "\n",  true);
for(int  i=0;i<headers.length();i++)
{
if(i==headers.length()-1)
break;
string  current_header=string_to_lower_case(headers[i]);
if(current_header=="content-length")
{
uint  size=string_to_number(headers[i+1]);
alert("Size", "The file is " + size + " bytes.");
exit();
}
}
}
alert("Error",  "The  size  of  the  file  could  not  be  retrieved.");
}

http object

This method initiates the retrieval of a URL from the Internet.

string post(string url, string parameters)

Parameters:
url
The URL of the file on the Internet that should be retrieved.
parameters
The parameters that should be passed to the server.

Return value:
A blank string on success, or an error description or blank string on failure (see remarks).

Remarks:
This method will send what is known as a POST request. A POST request is used in those cases where the data you are sending
is longer than is feasible with a GET request.

The URL may not contain any special characters such as spaces or similar. If you wish to include such characters, always
encode the URL by calling the encode function before attempting to retrieve it from the Internet.

If the http server to which the URL points uses a port other than 80, you will need to include the port after the host name
proceeded by a colon. The path on the server (if any) should then follow, proceeded by a slash. Such a URL might look like
http://myserver.com:8080/myfile.html.

If multiple requests are made with URL's pointing to the same server, the http object will attempt to keep the connection alive for
a short while which results in a dramatic speed increase for subsequent requests.

If the network subsystem fails to initialize, the get_last_error function will return -27 and the returned string will be blank.

If an Internet related error occurs, the get_last_error function will return -29 and the returned string will contain a textual
description of the error. This means that you should always call get_last_error after initiating a request.

This method can be used to retrieve both binary and textual data.

The http object is asynchronous, which means that it does not wait for any data before returning. To retrieve data from the
server, you must continuously call the request method until the progress boolean property reports false.

Example:
// Post some data to a dummy site.

void  main()
{
http  download;
string  body=download.post("http://www.mysite.com/test.php",  "Data=Test&ID=495&Name=Example");
if(get_last_error()!=0)
{
alert("Error",  "An  error  occured.\r\nDescription:  "  +  body  +  "");
}
else
{
while(download.progress)
{
body+=download.request();
wait(5);
}
clipboard_copy_text(body);

alert("Success",  "The  data  was  retrieved  and  has  been  copied  to  your  clipboard.");
}
}

http object

This method retrieves any new data from a currently active download using either the get or post methods.

string request()

Parameters:
None.

Return value:
The data from the server on success, or a blank string on failure.

Remarks:
This method checks if any new data has been returned by the server since the last call, and returns it as a string. It must be called
continuously, without any long pauses, if the retrieval is to succeed. You must keep calling this function until the progress boolean
property reports false.

Please note that it is perfectly normal for this method to return a blank string without any error having occured. This simply means
that no new data was provided by the server since the last call. Always call get_last_error to detect if an actual error has
occured.

Example:
//  Retrieve  the  contents  of  the  index  page  at  blastbay.com.

void  main()
{
http  download;
string  body=download.get("http://www.blastbay.com/");
if(get_last_error()!=0)
{
alert("Error",  "An  error  occured.\r\nDescription:  "  +  body  +  "");
}
else
{
while(download.progress)
{
body+=download.request();
wait(5);
}
clipboard_copy_text(body);
alert("Success",  "The  data  was  retrieved  and  has  been  copied  to  your  clipboard.");
}
}

http object

bool progress
This is set to true if an operation is in progress, and false if it has completed or if an error has occurred. This property cannot be
modified from the script.

string return_code
A string representing The server response which is given before data starts appearing. 200 ok represents success. This property
cannot be modified from the script.

string headers
A string representing the raw http headers returned by the server. These will normally not be needed unless you have some
specific reason for wanting to see them. This property cannot be modified from the script.

string user_agent
A string giving the browser name and version that is reported to the server from the client. The default is BGT/1.0. If you set it to
an empty string, it will be returned to the default again. This can be both set and retrieved.

instance object

The instance object is used to check whether a given game is already running.

instance(string application_name)

Parameters:
application_name
An optional unique name that is used internally for the game.

Remarks:
This object is useful if you wish to prevent multiple instances of the same game from being opened at the same time.

If the application_name argument is specified, it must be no more than 100 characters in length. If it is not specified, it will default to a hash of the game executable and hence will not work if trying to prevent an older version of a game from
running alongside an updated copy.

Note: This class will only work if the game is compiled.

Example:
/*
Write a script that checks for another instance of itself. To make this script work effectively you must run it at least twice within ten seconds of the first instance being started.
*/

void  main()
{
instance  the_game("instance_checker");
if(the_game.is_already_running)
{
alert("Error",  "This  script  is  already  running.");
exit();
}
wait(10000);
}

instance object

The instance object is used to check whether a given game is already running.

instance(string application_name)

Parameters:
application_name
An optional unique name that is used internally for the game.

Remarks:
This object is useful if you wish to prevent multiple instances of the same game from being opened at the same time.

If the application_name argument is specified, it must be no more than 100 characters in length. If it is not specified, it will default to a hash of the game executable and hence will not work if trying to prevent an older version of a game from
running alongside an updated copy.

Note: This class will only work if the game is compiled.

Example:
/*
Write a script that checks for another instance of itself. To make this script work effectively you must run it at least twice within ten seconds of the first instance being started.
*/

void  main()
{
instance  the_game("instance_checker");
if(the_game.is_already_running)
{
alert("Error",  "This  script  is  already  running.");
exit();
}
wait(10000);
}

instance object

bool is_already_running
Returns true if another instance is found to be running, false otherwise.

joystick object

The joystick object enables games to communicate with joysticks that are connected to the user's system.

joystick()

Parameters:
None.

Remarks:
When this object is constructed, the engine automatically attempts to open the joystick device that the user has set as their preferred one in the control panel. If you want the
user to be able to choose, use the list_joysticks method and then the set method. If you want to account for users unplugging and plugging in joysticks while the program is
running, call the refresh_joystick_list method and then call list_joysticks again before selecting a new one with the set method.

Joysticks generally support many different types of axies. To make calculations easier, all of these have been set to have a range between -1000 and 1000. See the
properties chapter for more details on what axies are available and what each one does. Note that forward movement of any y axis on a stick will be represented by a
negative number, and backward movement by a positive number. This is in accordance with how the same movement would be shown on a graph.

Please note: The joystick object currently does not support force feedback.

Example:
//  This  is  a  simple  and  rather  unrealistic  truck  simulation.

/*
This program plays an engine sound and allows the user to move it around with a joystick. Escape is pressed to close the program.
*/

void  main()
{
show_game_window("Joy  Truck");
joystick  stick;
if(stick.active==false)
{
alert("Error",  "No  joystick  seems  to  be  attached.");
exit();
}
sound  truck;
truck.load("engine.wav");
if(truck.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
truck.play_looped();
while(key_pressed(KEY_ESCAPE)==false)
{

// Retrieve the current coordinates of the stick. Remember, the range is between -1000 and 1000.
double  x=stick.x;
double  y=stick.y;

/*
Now we have to scale these absolute values to be suitable for pan and pitch.
X  represents  pan,  and  y  represents  pitch.
Note that for the pitch to increase when the stick is pushed forward, we must reverse y.
If it is positive it must be made negative, and if it is negative it must be made positive.
*/

// By dividing x by 25, we get a range that is between -40 and 40.
x/=25;

//  Reverse  the  Y  coordinate.
if(y<0)

y=absolute(y);
else
y=0-y;

// Now put y into the positive range (between 0 and 2000, and then divide it by 10.
y+=1000;
y/=10;

// We don't want the pitch to go lower than 20.
if(y<20)
y=20;

// Now we have our values, so we can assign them to their respective properties in our truck sound object.
truck.pan=x;
truck.pitch=y;
wait(5);
}
}

joystick object

The joystick object enables games to communicate with joysticks that are connected to the user's system.

joystick()

Parameters:
None.

Remarks:
When this object is constructed, the engine automatically attempts to open the joystick device that the user has set as their preferred one in the control panel. If you want the
user to be able to choose, use the list_joysticks method and then the set method. If you want to account for users unplugging and plugging in joysticks while the program is
running, call the refresh_joystick_list method and then call list_joysticks again before selecting a new one with the set method.

Joysticks generally support many different types of axies. To make calculations easier, all of these have been set to have a range between -1000 and 1000. See the
properties chapter for more details on what axies are available and what each one does. Note that forward movement of any y axis on a stick will be represented by a
negative number, and backward movement by a positive number. This is in accordance with how the same movement would be shown on a graph.

Please note: The joystick object currently does not support force feedback.

Example:
//  This  is  a  simple  and  rather  unrealistic  truck  simulation.

/*
This program plays an engine sound and allows the user to move it around with a joystick. Escape is pressed to close the program.
*/

void  main()
{
show_game_window("Joy  Truck");
joystick  stick;
if(stick.active==false)
{
alert("Error",  "No  joystick  seems  to  be  attached.");
exit();
}
sound  truck;
truck.load("engine.wav");
if(truck.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
truck.play_looped();
while(key_pressed(KEY_ESCAPE)==false)
{

// Retrieve the current coordinates of the stick. Remember, the range is between -1000 and 1000.
double  x=stick.x;
double  y=stick.y;

/*
Now we have to scale these absolute values to be suitable for pan and pitch.
X  represents  pan,  and  y  represents  pitch.
Note that for the pitch to increase when the stick is pushed forward, we must reverse y.
If it is positive it must be made negative, and if it is negative it must be made positive.
*/

// By dividing x by 25, we get a range that is between -40 and 40.
x/=25;

//  Reverse  the  Y  coordinate.
if(y<0)

y=absolute(y);
else
y=0-y;

// Now put y into the positive range (between 0 and 2000, and then divide it by 10.
y+=1000;
y/=10;

// We don't want the pitch to go lower than 20.
if(y<20)
y=20;

// Now we have our values, so we can assign them to their respective properties in our truck sound object.
truck.pan=x;
truck.pitch=y;
wait(5);
}
}

joystick object

This function checks if a particular button on the joystick is currently being held down.

bool button_down(int button)

Parameters:
button
The ID of a joystick button to check (see remarks).

Return value:
true if the button is held down, false if it is not or if an error occurs.

Remarks:
The difference between button_down and button_pressed is that button_pressed will only return true when the user first pushes down the button, while button_down will continue returning true until the button is released again.

The joystick object supports up to 128 buttons, ranging from 0 to 127. However the actual maximum button is determined by the buttons property for the given device, ranging from 0 to buttons minus 1.

It is important to remember when mapping functions to joystick buttons that devices can vary greatly between models, supporting anything between 3 and 15 buttons, and therefore it is always a good idea to map the most common activities to buttons with
lower numbers.

Example:
// The program will exit when the user holds down button 1 and presses button 2. Of course, we shall allow escape as well, just in case this combination should prove difficult with some models.

void  main()
{
joystick  stick;
show_game_window("Joystick  Test");
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
while(!key_pressed(KEY_ESCAPE))
{
if(stick.button_down(1)&&(stick.button_pressed(2)))
{
exit();
}
}
}

joystick object

This function checks if a particular button on the joystick has been pressed.

bool button_pressed(int button)

Parameters:
button
The ID of a joystick button to check (see remarks).

Return value:
true if the button has been pressed, false if it has not or if an error occurs.

Remarks:
The difference between button_down and button_pressed is that button_pressed will only return true when the user first pushes
down the button, while button_down will continue returning true until the button is released again.

The joystick object supports up to 128 buttons, ranging from 0 to 127. However the actual maximum button is determined by
the buttons property for the given device, ranging from 0 to buttons minus 1.

It is important to remember when mapping functions to joystick buttons that devices can vary greatly between models,
supporting anything between 3 and 15 buttons, and therefore it is always a good idea to map the most common activities to
buttons with lower numbers.

Example:
// Play a looping sound whenever a button is pressed.

void  main()
{
sound  loop;
joystick  stick;
int  current_loop=-1;
show_game_window("Joystick  Test");
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
while(!key_pressed(KEY_ESCAPE))
{
for(int  counter=0;  counter<stick.buttons;  counter++)
{
if(stick.button_pressed(counter))
{
if(current_loop!=counter)
{
loop.load("loop"+counter+".wav");
current_loop=counter;
}
loop.play_looped();
}
if(stick.button_released(counter))
{
loop.stop();
}
}
}
}

joystick object

This function checks if a particular button on the joystick has been released.

bool button_released(int button)

Parameters:
button
The ID of a joystick button to check (see remarks).

Return value:
true if the button has been released, false if it has not or if an error occurs.

Remarks:
The difference between button_up and button_released is that button_released will only return true when the user first releases the
button, while button_up will continue returning true until the button is pushed down again.

The joystick object supports up to 128 buttons, ranging from 0 to 127. However the actual maximum button is determined by the
buttons property for the given device, meaning that the allowed range is from 0 to buttons minus 1.

It is important to remember when mapping functions to joystick buttons that devices can vary greatly between models, supporting
anything between 3 and 15 buttons, and therefore it is always a good idea to map the most common activities to buttons with
lower numbers.

Example:
// Play a looping sound whenever a button is pushed down, stopping it when the button is released.

void  main()
{
sound  loop;
joystick  stick;
int  current_loop=-1;
show_game_window("Joystick  Test");
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
while(!key_pressed(KEY_ESCAPE))
{
for(int  counter=0;  counter<stick.buttons;  counter++)
{
if(stick.button_pressed(counter))
{
if(current_loop!=counter)
{
loop.load("loop"+counter+".wav");
current_loop=counter;
}
loop.play_looped();
}
if(stick.button_released(counter))
{
loop.stop();
}
}
}
}

joystick object

This function checks if a particular button on the joystick is currently up.

bool button_up(int button)

Parameters:
button
The ID of a joystick button to check (see remarks).

Return value:
true if the button is up, false if it is not or if an error occurs.

Remarks:
The difference between button_up and button_released is that button_released will only return true when the user first releases
the button, while button_up will continue returning true until the button is pushed down again.

The joystick object supports up to 128 buttons, ranging from 0 to 127. However the actual maximum button is determined by
the buttons property for the given device, meaning that the allowed range is from 0 to buttons minus 1.

It is important to remember when mapping functions to joystick buttons that devices can vary greatly between models,
supporting anything between 3 and 15 buttons, and therefore it is always a good idea to map the most common activities to
buttons with lower numbers.

Example:
// The engine sound will only play when button 0 is down.

void  main()
{
show_game_window("Joystick  buttons");
sound  truck;
joystick  stick;
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
truck.load("engine.wav");
while(key_up(KEY_ESCAPE))
{
if((stick.button_up(0))&&(truck.playing))
truck.pause();
else
truck.play_looped();
wait(5);
}
}

joystick object

This function returns a list of all the buttons on the joystick that are currently being held down.

int[] buttons_down()

Parameters:
None.

Return value:
An array with the currently held buttons, or an empty array if no buttons are being held down or if an error occurs.

Remarks:
The difference between buttons_down and buttons_pressed is that buttons_pressed will only return a list of the buttons that the
user has just pushed down, while buttons_down will return a list of the buttons that are currently being held down regardless of
whether the same buttons have been reported as being down in a previous call.

The joystick object supports up to 128 buttons, ranging from 0 to 127. However the actual maximum button is determined by
the buttons property for the given device, ranging from 0 to buttons minus 1.

It is important to remember when mapping functions to joystick buttons that devices can vary greatly between models,
supporting anything between 3 and 15 buttons, and therefore it is always a good idea to map the most common activities to
buttons with lower numbers.

Example:
// Wait for three different buttons to be held down at once before closing the program.

void  main()
{
show_game_window("buttons  Down  Test");
int[]  button_list;
joystick  stick;
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
while(button_list.length()<3)
{
button_list=stick.buttons_down();
wait(5);
}
}

joystick object

This function returns a list of all the buttons on the joystick that have just been pressed.

int[] buttons_pressed()

Parameters:
None.

Return value:
An array with the buttons that have just been pressed, or an empty array if no buttons have been pressed since the last call or if
an error occurs.

Remarks:
The difference between buttons_down and buttons_pressed is that buttons_pressed will only return a list of the buttons that the
user has just pushed down, while buttons_down will return a list of the buttons that are currently being held down regardless of
whether the same buttons have been reported as being down in a previous call.

Please note that buttons_pressed and button_pressed use the same joystick state, so if a call to buttons_pressed is made right
before a call to button_pressed, none of the buttons returned by buttons_pressed will be reported as just having been pushed
down by button_pressed. The same holds true for the reverse senario.

The joystick object supports up to 128 buttons, ranging from 0 to 127. However the actual maximum button is determined by
the buttons property for the given device, meaning that the allowed range is from 0 to buttons minus 1.

It is important to remember when mapping functions to joystick buttons that devices can vary greatly between models,
supporting anything between 3 and 15 buttons, and therefore it is always a good idea to map the most common activities to
buttons with lower numbers.

Example:
// Speak the button codes of any buttons that are pressed, and exit if the user presses escape.

void  main()
{
show_game_window("buttons  Pressed  Test");
tts_voice  speech;
joystick  stick;
int[]  list;
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
while(key_pressed(KEY_ESCAPE)==false)
{
list=stick.buttons_pressed();
for(int  i=0;i<list.length();i++)
{
if(i==0)
{
speech.speak_interrupt(list[i]);
}
else
{
speech.speak(list[i]);
}
}
wait(5);
}
}

joystick object

This function returns a list of all the buttons on the joystick that have just been released.

int[] buttons_released()

Parameters:
None.

Return value:
An array with the buttons that have just been released, or an empty array if no buttons have been released since the last call or if
an error occurs.

Remarks:
The difference between buttons_up and buttons_released is that buttons_released will only return a list of the buttons that the
user has just released, while buttons_up will return a list of the buttons that are currently up regardless of whether the same
buttons have been reported as being up in a previous call.

Please note that buttons_released and button_released use the same joystick state, so if a call to buttons_released is made right
before a call to button_released, none of the buttons returned by buttons_released will be reported as just having been released
by button_released. The same holds true for the reverse senario.

The joystick object supports up to 128 buttons, ranging from 0 to 127. However the actual maximum button is determined by
the buttons property for the given device, ranging from 0 to buttons minus 1.

It is important to remember when mapping functions to joystick buttons that devices can vary greatly between models,
supporting anything between 3 and 15 buttons, and therefore it is always a good idea to map the most common activities to
buttons with lower numbers.

Example:
// Speak the button codes of any buttons that are released, and exit if the user presses escape.

void  main()
{
show_game_window("buttons  released  Test");
tts_voice  speech;
joystick  stick;
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
int[]  list;
while(key_pressed(KEY_ESCAPE)==false)
{
list=stick.buttons_released();
for(int  i=0;i<list.length();i++)
{
if(i==0)
{
speech.speak_interrupt(list[i]);
}
else
{
speech.speak(list[i]);
}
}
wait(5);
}
}

joystick object

This function returns a list of all the buttons on the joystick that are currently up.

int[] buttons_up()

Parameters:
None.

Return value:
An array with the buttons that are up, or an empty array if no buttons are up or if an error occurs.

Remarks:
The difference between buttons_up and buttons_released is that buttons_released will only return a list of the buttons that the user has just released, while buttons_up will
return a list of the buttons that are currently up regardless of whether the same buttons have been reported as being up in a previous call.

The joystick object supports up to 128 buttons, ranging from 0 to 127. However the actual maximum button is determined by the buttons property for the given device,
ranging from 0 to buttons minus 1.

It is important to remember when mapping functions to joystick buttons that devices can vary greatly between models, supporting anything between 3 and 15 buttons, and
therefore it is always a good idea to map the most common activities to buttons with lower numbers.

Example:
// The example will constantly monitor buttons that are up. If it registers three or more buttons that are not up, it will exit.

void  main()
{
show_game_window("buttons  up  Test");
int[]  button_list;
joystick  stick;
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
button_list=stick.buttons_up();
while(button_list.length()>125)
{
button_list=stick.buttons_up();
wait(5);
}
}

joystick object

This method returns an array containing the names of all connected and available joysticks on the user's system.

string[] list_joysticks()

Parameters:
None.

Return value:
An array containing the names of all available joysticks on success, or an empty list on failure or if no joysticks are available.

Remarks:
The entries in this list are useful if you wish to give the user a choice as to what joystick that should be used. You may use an index in this list
when calling the set method to activate a given joystick.

Please note that the contents of this list are retrieved from a cache, as opposed to being reenumerated each time this method is called. This is to
prevent joystick ID's from being invalidated if the user should plug out an existing joystick or plug in a new one during the execution of your
game. To refresh this list, call the refresh_joystick_list method.

Example:
//  Enumerate  the  list  of  joysticks  and  display  information  on  them.

void  main()
{
joystick  stick;
show_game_window("Joystick  Test");
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
string[]  names=stick.list_joysticks();
for(int  i=0;i<names.length();i++)
{
alert("Joystick  "  +  (i+1),  names[i]);
stick.set(i);
if(stick.active==false)
{
alert("Error",  "Could  not  open  the  joystick  device.");
continue;
}
alert("Information",  "Buttons:  "  +  stick.buttons  +  "\nSliders:  "  +  stick.sliders  +  "\nPovs:  "  +  stick.povs);
}
}

joystick object

This method checks whether a particular pov is centered on the joystick.

bool pov_centered(int pov)

Parameters:
pov
The index of the pov to check (see remarks).

Return value:
true if the selected pov is centered, false if not or if an error has occurred.

Remarks:
The joystick object supports up to four povs, ranging from 1 to 4, though the actual maximum depends on the povs property for the given device.

Example:
// Check each pov on the joystick to see if it is centered. Wait two seconds for the script to work out what is central when it opens.

void  main()
{
joystick  stick;
show_game_window("Joystick  Test");
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
show_game_window("Pov  Center  Example");
wait(2000);
for(int  counter=1;  counter<=stick.povs;  counter++)
{
if(stick.pov_centered(counter))
{
alert("pov",  "pov  "+counter+"  centred.");
}
}
}

joystick object

This method refreshes the list of available joysticks.

bool refresh_joystick_list()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
The joystick list is cached in order to prevent the ID's becoming invalid. This method reenumerates all the joysticks, and also automatically closes the currently open joystick device if any. It is necessary to call the set
method with an ID after reenumerating the joysticks.

Example:
// Refresh the list of joysticks continuously, and print a message if the list changes.

joystick  stick;

timer  recheck;

void  main()
{
show_game_window("Joystick  searcher");
alert("Information",  "There  are  currently  "+stick.joysticks+"  joysticks  connected.  Monitoring...");
string[]  stick_list=stick.list_joysticks();
recheck.restart();
while(key_pressed(KEY_ESCAPE)==false)
{
if(recheck.elapsed>=1000)
{
stick.refresh_joystick_list();
recheck.restart();
string[]  new_list=stick.list_joysticks();
if(new_list.length()==stick_list.length())
continue;
if(new_list.length()<stick_list.length())
{
alert("Terror!",  "One  or  more  joysticks  have  now  become  sleeping  sadsticks  as  they  were  disconnected  from  the  system.");
}
else
{
alert("Joy!", "One or more new joysticks are now living up to their name as they are now active, connected to the system, and happily waiting for something to do!");
}
stick_list=new_list;
}
wait(5);
}
}

joystick object

This method explicitly opens a joystick device on the user's system.

bool set(int index)

Parameters:
index
The index of the device to open (see remarks).

Return value:
true on success, false on failure.

Remarks:
The index corresponds to the list returned by list_joysticks, and is zero-based, ranging from 0 to joysticks minus 1.

If no joystick is explicitly opened, the engine will automatically open the user's preferred joystick as set in control panel. This is done when the
joystick object instance is first created.

Example:
//  Enumerate  the  list  of  joysticks  and  display  information  on  the  first  available.

void  main()
{
joystick  stick;
show_game_window("Joystick  Test");
if(stick.joysticks==0)
{
alert("Error",  "No  joysticks  seem  to  be  attached.");
exit();
}
string[]  names=stick.list_joysticks();
for(int  i=0;i<names.length();i++)
alert("Joystick  "  +  (i+1),  names[i]);
stick.set(0);
if(stick.active==false)
{
alert("Error",  "Could  not  open  the  joystick  device.");
exit();
}
alert("Information",  "Buttons:  "  +  stick.buttons  +  "\nSliders:  "  +  stick.sliders  +  "\nPovs:  "  +  stick.povs);
}

joystick object

uint joysticks
The number of joysticks that are currently attached. This property cannot be modified from the script.

bool has_x
Checks to see if the joystick has an x axis. This property cannot be modified from the script.

bool has_y
Checks to see if the joystick has a y axis. This property cannot be modified from the script.

bool has_z
Checks to see if the joystick has a z axis. This property cannot be modified from the script.

bool has_r_x
Checks to see if the joystick has a rotational x axis. This property cannot be modified from the script.

bool has_r_y
Checks to see if the joystick has a rotational y axis. This property cannot be modified from the script.

bool has_r_z
Checks to see if the joystick has a rotational z axis. This property cannot be modified from the script.

uint buttons
The number of buttons on the device. This property cannot be modified from the script.

uint sliders
The number of sliders on the device. This property cannot be modified from the script.

uint povs
The number of povs (point of view controls) on the device. This property cannot be modified from the script.

string name
The name of the device that is currently being used. This property cannot be modified from the script.

bool active
This is true if a joystick device is opened, false if not. This property cannot be modified from the script.

int preferred_joystick
Returns the index of the joystick that the user has set as their preferred device in the control panel. This corresponds to the list
returned by list_joysticks, and is never greater than the joysticks property -1. On error or if there is no default, -1 is returned.
This property cannot be modified in the script.

long x
X-axis, usually the left-right movement of a stick. This property cannot be modified from the script.

long y
Y-axis, usually the forward-backward movement of a stick. This property cannot be modified from the script.

long z
Z-axis, often the throttle control. If the joystick does not have this axis, the value is 0. This property cannot be modified from the
script.

long r_x

X-axis rotation. If the joystick does not have this axis, the value is 0. This property cannot be modified from the script.

long r_y
Y-axis rotation. If the joystick does not have this axis, the value is 0. This property cannot be modified from the script.

long r_z
Z-axis rotation (often called the rudder). If the joystick does not have this axis, the value is 0. This property cannot be modified
from the script.

long slider_1
An additional axis value (formerly called the u-axis) whose semantics depend on the joystick. This property cannot be modified
from the script.

long slider_2
An additional axis value (formerly called the v-axis) whose semantics depend on the joystick. This property cannot be modified
from the script.

uint pov_1
Direction controller, such as point-of-view hat. The position is indicated in hundredths of a degree clockwise from north (away
from the user). The center position is normally reported as - 1, but not all devices follow this rule. To check if a POV is centered,
use the pov_centered method. For indicators that have only five positions, the value for a controller is - 1, 0, 9,000, 18,000, or
27,000. This property cannot be modified from the script.

uint pov_2
Direction controller, such as point-of-view hat. The position is indicated in hundredths of a degree clockwise from north (away
from the user). The center position is normally reported as - 1, but not all devices follow this rule. To check if a POV is centered,
use the pov_centered method. For indicators that have only five positions, the value for a controller is - 1, 0, 9,000, 18,000, or
27,000. This property cannot be modified from the script.

uint pov_3
Direction controller, such as point-of-view hat. The position is indicated in hundredths of a degree clockwise from north (away
from the user). The center position is normally reported as - 1, but not all devices follow this rule. To check if a POV is centered,
use the pov_centered method. For indicators that have only five positions, the value for a controller is - 1, 0, 9,000, 18,000, or
27,000. This property cannot be modified from the script.

uint pov_4
Direction controller, such as point-of-view hat. The position is indicated in hundredths of a degree clockwise from north (away
from the user). The center position is normally reported as - 1, but not all devices follow this rule. To check if a POV is centered,
use the pov_centered method. For indicators that have only five positions, the value for a controller is - 1, 0, 9,000, 18,000, or
27,000. This property cannot be modified from the script.

long v_x
The X-axis velocity. This property cannot be modified from the script.

long v_y
Y-axis velocity. This property cannot be modified from the script.

long v_z
Z-axis velocity. This property cannot be modified from the script.

long vr_x
X-axis angular velocity. This property cannot be modified from the script.

long vr_y
Y-axis angular velocity. This property cannot be modified from the script.

long vr_z
Z-axis angular velocity. This property cannot be modified from the script.

long v_slider_1
Extra axis velocity. This property cannot be modified from the script.

long v_slider_2
Extra axis velocity. This property cannot be modified from the script.

long a_x
X-axis acceleration. This property cannot be modified from the script.

long a_y
Y-axis acceleration. This property cannot be modified from the script.

long a_z
Z-axis acceleration. This property cannot be modified from the script.

long ar_x
X-axis angular acceleration. This property cannot be modified from the script.

long ar_y
Y-axis angular acceleration. This property cannot be modified from the script.

long ar_z
Z-axis angular acceleration. This property cannot be modified from the script.

long a_slider_1
Extra axis acceleration. This property cannot be modified from the script.

long a_slider_2
Extra axis acceleration. This property cannot be modified from the script.

long f_x
X-axis force. This property cannot be modified from the script.

long f_y
Y-axis force. This property cannot be modified from the script.

long f_z
Z-axis force. This property cannot be modified from the script.

long fr_x
X-axis torque. This property cannot be modified from the script.

long fr_y
Y-axis torque. This property cannot be modified from the script.

long fr_z
Z-axis torque. This property cannot be modified from the script.

long f_slider_1
Extra axis force. This property cannot be modified from the script.

long f_slider_2
Extra axis force. This property cannot be modified from the script.

library object

The library object is used to call functions stored in an external dynamic link library (DLL) file, such as those implementing the Windows API.

library()

Parameters:
None.

Remarks:
Please note: The library object is still in an early stage, and should be considered beta. Use at your own risk. This does not apply to the rest of the features in this version of BGT,
which are production quality. Note also that there are several missing features in the library interface, such as the ability to use C style structures. This will be addressed as soon as
possible. This documentation will also be extended with a much more thorough description/tutorial.

When a library object is first created, it will not be active. To activate it you must call the "load" method on the object, specifying the filename of a DLL that should be associated with
it.

When using this object, it is essential to exercise extra caution and to thoroughly read the reference for the library you are trying to access, since using this object incorrectly may
cause the application to crash. It may also cause memory leaks and/or memory corruption, which would lead to very hard to find errors that would eventually cause crashes in a
seemingly random location later.

Example:
//  Show  a  message  box  using  the  conventional  Windows  MessageBox  function.

const  int  win32_mb_icon_information=0x00000040;  //  The  value  as  written  in  the  MSDN.

void  main()
{
library  dll;
dll.load("user32.dll");

// The MessageBox parameter takes a window handle, the text, the title, and the flags.
dll.call("int  MessageBoxA(int,  char*,  char*,  int);",  0,  "I  am  a  message  box  using  user32.dll.",  "Hey  there",  win32_mb_icon_information);
}

library object

The library object is used to call functions stored in an external dynamic link library (DLL) file, such as those implementing the Windows API.

library()

Parameters:
None.

Remarks:
Please note: The library object is still in an early stage, and should be considered beta. Use at your own risk. This does not apply to the rest of the features in this version of BGT,
which are production quality. Note also that there are several missing features in the library interface, such as the ability to use C style structures. This will be addressed as soon as
possible. This documentation will also be extended with a much more thorough description/tutorial.

When a library object is first created, it will not be active. To activate it you must call the "load" method on the object, specifying the filename of a DLL that should be associated with
it.

When using this object, it is essential to exercise extra caution and to thoroughly read the reference for the library you are trying to access, since using this object incorrectly may
cause the application to crash. It may also cause memory leaks and/or memory corruption, which would lead to very hard to find errors that would eventually cause crashes in a
seemingly random location later.

Example:
//  Show  a  message  box  using  the  conventional  Windows  MessageBox  function.

const  int  win32_mb_icon_information=0x00000040;  //  The  value  as  written  in  the  MSDN.

void  main()
{
library  dll;
dll.load("user32.dll");

// The MessageBox parameter takes a window handle, the text, the title, and the flags.
dll.call("int  MessageBoxA(int,  char*,  char*,  int);",  0,  "I  am  a  message  box  using  user32.dll.",  "Hey  there",  win32_mb_icon_information);
}

library object

This method will call a specified function from within the associated DLL library, returning a handle to a dictionary containing the results of the call.

dictionary@ call(string function_signature, ? list_of_parameters)

Parameters:
function_signature
The function signature expected by the library.
list_of_parameters
The list of parameters expected by the specified function signature.

Return value:
A handle to a dictionary containing the results of the call. If the call failed, the dictionary will be empty (see remarks).

Remarks:
Similarly to calling functions in BGT, the list of parameters are passed according to the function signature provided in the first parameter, and are separated by commas.

If the call was successful, the function will return a handle to a dictionary containing all the results of the call. Since some parameters may actually be modified by the library, I.E. if they are passed as
pointers (see below), the dictionary will provide the return value as key 0, and the parameters we passed to the DLL as the number of the parameter (see the example for how this works in practice).

At present, it is only possible to call DLL's that export a C style interface either using stdcall or cdecl, which the absolute majority of them do. All the Windows API DLL's, for instance, use stdcall
and export C interfaces.

The function signature you provide should match that stored in the library for it to succeed.

Below is a list of datatypes that the function signature can currently process.

char*: Equivalent to a BGT string. Internally, this is a pointer to an array of characters (see below).
char: The same as int8 in BGT.
unsigned char: The same as uint8 in BGT.
short, int, long: The same as in BGT.
unsigned short, unsigned int, unsigned long: The same as ushort, uint and ulong in BGT.
BOOL: This is a Windows datatype that is the same size as a long.
Handles: The Windows API has several different handle datatypes. These can all be processed as an int, as they are pointers. If a handle needs to be passed as null, the null keyword can be
used in the same way as in BGT.

Sometimes, DLL's may expect and/or return what are called pointers. Put simply, a pointer is a special type of variable in C which provides the location in memory of a particular piece of data. If you
are familiar with object handles in BGT, then you can apply a similar concept to pointers in C. If a function expects a pointer, then you give it the variable type, followed by a star sign (*). For
example, to pass a pointer to an int you would use int*. To pass a string, you would use char*, since a string is a pointer to an array of characters.

if a parameter you are passing is smaller than the type that the function expects based on the signature string, it will be converted if possible. For instance, let's say the function takes an int and you
pass a short from the script, it will automatically be converted to an int internally before it is passed. However, implicit type conversions are only attempted if the parameter in question is not a pointer.

Theoretically, function signatures can contain parameters with multiple pointers, such as int**, int*** etc. At present, only one level of indirection is supported. This means that only one pointer, such
as int*, can be used for any one parameter.

If for some reason the call fails the parsing sequence, a runtime error will be triggered.

Example:
// Retrieve the computer name and display it in a message box, all with DLL's.

//  Define  the  information  constant.
const  int  win32_mb_icon_information=0x00000040;

void  main()

{

//  Declare  some  variables.
library dll; // This will do all the nifty work.
dictionary@  dll_result;  //  This  will  contain  the  result  of  the  call.
string name; // This will hold our computer name.

/*
Since the DLL expects us to allocate our own memory for the string, we will resize the string to a decent length so that the DLL can fill it.
*/
name.resize(500);

//  Load  the  library.
dll.load("kernel32.dll");

// Call the function that will retrieve our computer name. It takes the string that is to hold the computer name, and its length.
@dll_result=dll.call("int  GetComputerNameA(char*,  int*);",  name,  name.length());

//  Unload  the  library.
dll.unload();

// Retrieve the modified computer name. Since the string that we passed was the first parameter, we retrieve this with key 1.
dll_result.get("1",  name);

//  Remove  the  remaining  null  characters.
name=string_replace(name,  "\0",  "",  true);
// Now we create a message box with the information.

//  Load  the  library  containing  the  MessageBox  function.
dll.load("user32.dll");

//  Call  the  MessageBox  function.
dll.call("int  MessageBoxA(int,  char*,  char*,  int);",  0,  "The  name  assigned  to  this  computer  is  "+name+".",  "Information",  win32_mb_icon_information);
}

Library object

This method will load a DLL library.

bool load(string library_filename)

Parameters:
library_filename
The filename of the DLL to open, with or without extension.

Return value:
true on success, false on failure.

Remarks:
When specifying the filename, either an absolute or relative path can be used. If no absolute path is specified, the DLL will be searched for in the script directory, followed by the
system directory, followed by the Windows directory. If the DLL cannot be found in any of these locations, the method will return false.

Example:
//  Show  a  message  box  using  the  conventional  Windows  MessageBox  function.

const  int  win32_mb_icon_information=0x00000040;  //  The  value  as  written  in  the  MSDN.

void  main()
{
library  dll;
dll.load("user32.dll");

// The MessageBox parameter takes a window handle, the text, the title, and the flags.
dll.call("int  MessageBoxA(int,  char*,  char*,  int);",  0,  "I  am  a  message  box  using  user32.dll.",  "Hey  there",  win32_mb_icon_information);
}

library object

This method will unload the library currently associated with the object.

bool unload()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
It is essential that you free any memory or other resources that the DLL may be using before unloading the library. Failure to do this could cause memory leaks or even crashes. Check the DLL's reference to find out whether this is necessary and for instructions on how to do this.

Example:
//  This  code  will  not  work  unless  you  have  a  library  called  my_recorder.dll  containing  the  functions  that  this  script  will  attempt  to  execute.  This  merely  demonstrates  the  necessity  for  some  libraries  to  have  their  associated  resources  cleared  before  unloading  them.

//  Declare  some  variables.
library  dll;
dictionary@  dll_params;

void  main()
{
record("test.wav");
}

void  record(string  filename)
{
int  result=-1;  //  For  storing  return  values  from  the  library.
int  handle=-1;  //  For  manipulating  a  file.

//  Load  the  library.
dll.load("my_recorder.dll");

//  Initialise  the  library.
@dll_params=dll.call("int  initialise_recorder();");
if(dll_params.is_empty())
{
alert("Error",  "Failed  to  call  function  initialise_recorder.");
exit();
}
dll_params.get("0",  result);
if(result!=1)
{
alert("Error",  "Recorder  initialisation  failed.");
exit();
}

// Open a file.
@dll_params=dll.call("int  create_file(char*);",  filename);
if(dll_params.is_empty())
{
alert("Error",  "File  could  not  be  created.");
cleanup();
exit();
}
dll_params.get("0",  handle);
if(handle==-1)
{
alert("Error",  "File  creation  failed.");
cleanup();
exit();
}

// Record to the file.
@dll_params=dll.call("int  record(int);",  handle);
if(dll_params.is_empty())
{
alert("Error",  "Cannot  record  to  file.");
cleanup(handle);
file_delete(filename);
exit();
}
dll_params.get("0",  result);
if(result!=1)
{
alert("Error",  "Recording  failed.");
cleanup(handle);
file_delete(filename);
exit();
}

// Wait for five seconds to allow a recording to form.
wait(5000);

//  Stop  recording.
@dll_params=dll.call("int  stop(int);",  handle);
if(dll_params.is_empty())
{
cleanup(handle);
exit();
}
dll_params.get("0",  result);
if(result!=1)
{
cleanup(handle);
exit();
}

//Perform  cleanup.
cleanup(handle);
}

void  cleanup(int  handle=-1)
{
int  result=-1;

//  See  if  we  need  to  close  the  file.
if(handle>-1)
{
@dll_params=dll.call("int  Close(int);",  handle);
if(dll_params.is_empty())
{
alert("Oops!",  "Cannot  close  file.");
cleanup();
exit();
}
dll_params.get("0",  result);
if(result!=1)
{
alert("Ouch!",  "File  close  operation  failed!");
cleanup();
exit();
}
}

//  Free  the  recorder.
@dll_params=dll.call("int  free_recorder();");
if(dll_params.is_empty())
{
alert("Terror!",  "Unable  to  free  the  recorder  library!");
exit();
}

dll_params.get("0",  result);
if(result!=1)
{
alert("Horror!",  "Recorder  cleanup  procedure  failed!");
exit();
}
dll.unload();
}

library object

bool active
This is set to true when a DLL is associated with the object, and to false otherwise. This property cannot be modified from the
script.

network object

The network object is used to allow games to be played online with other live players in a peer to peer environment.

network()

Parameters:
None.

Remarks:
This object allows the creation of multiplayer games of any complexity. One player will usually act as a server while the rest connect to that server.

The procedure when writing a networked game in BGT is to either set up a server and then wait for incoming connections, or to connect to a remote computer using its IP address and a designated port. Port numbers range from 1 to 65535, however most ports below 1000 are reserved and should thus be avoided. A server must be listening on the same port as the client uses when connecting
to the server, in order for a connection to be successfully established.

There are two components to the networking system in BGT. There is the network object itself, and an object called network_event, which contains information about a received event. This network event is filled in with information when a call is made to the request method, and you then examine the properties of this returned object in order to determine what has occured.

As soon as you start using the network object, you must continuously call the request method in order to get events from the outside world. A server will receive an event when a new person has connected, for instance, and a client will receive an event when an attempted connection to a server has either succeeded or failed.

Along with each event, you will get a so called peer ID. A peer ID is a unique number that identifies a peer. When a client attempts to connect to a server, for example, the connect method will return a peer ID immediately. This peer ID will then be in the process of connecting, and will not be active until a connect event has been received which contains the same ID. Similarly, when a server
receives a new incoming connection event, this event will always contain a peer ID so that the server is able to uniquely identify this peer when sending data to and receiving data from it.

Once a connection has been successfully established between two or more players, these players will begin passing data back and forth between one another by sending so called packets. A packet is basically a sequence of bytes given as a string. The packets will always arrive in the same order in which they were sent, meaning that you will never get an older packet after a newer one has
already been delivered. You can send packets in one of two ways, reliably and unreliably.

Unreliable packets are by far the quickest way of getting data across to another person. An unreliable packet will be sent at the maximum speed provided by the connection, but is not guaranteed to arrive at the destination. It is, however, guaranteed to be properly sequenced if it does arrive. Unreliable packets are useful when you have data that needs to be sent out very frequently, but you only
really care about the most recent data. You are not particularly interested in another player's position from 2 seconds ago, for instance, but rather you want to know where they are at this very moment.

Reliable packets are always guaranteed to get through, and always in the proper order just like unreliable ones. This reliability comes at a cost, however. Since the internet is unreliable by nature, BGT needs to wait for an acknowledgement from the other person to say that the packet was in fact received. If no such acknowledgement is received within a reasonable amount of time, a new attempt
to send the packet is made. The same procedure is repeated over and over again until an acknowledgement is received, or until enough time has elapsed to deem the connection as dead. This may cause considerable delays in the transmition as compared to unreliable packets, and should thus only be used for things where speed is not critical.

A problem arises when trying to send both unreliable and reliable packets at the same time. Since packets are required to be sequenced and reliable packets require verification before they are delivered, they will stall any unreliable packets that come after it in the sequence. Then, once the reliable packet is received the unreliable ones will arrive all at once. To prevent this, BGT uses channels. A
channel is an independently sequenced queue of packets, meaning that you can send both reliable and unreliable packets at the same time on different channels without them blocking one another. Channels also allow you to structure your data more efficiently, by for example sending position updates on one channel and shooting events on another.

Care should be taken when using multiple channels, to avoid a situation where one channel lags behind the others in time due to multiple reliable packets being stalled for example. Thus, if sequencing is important for a series of events, it may be necessary to put them on the same channel and wait for them to actually get through. Different channels are only beneficial when the usefulness of one
packet is not strictly dependent on another packet having been delivered before it.

Channels are 0 based, meaning that if you request a connection with 3 channels you may use channels 0, 1, and 2 when sending your data.

When disconnecting from a remote peer, you have three methods to choose from. disconnect_peer_forcefully, disconnect_peer, and disconnect_peer_softly. More information about how each method works can be found in that method's chapter, but the choice of method all has to do with how and when you want the remote peer to be notified of your disconnection. After calling
disconnect_peer or disconnect_peer_softly, you must keep checking for events on the host as the disconnection will not be complete until a disconnect event containing the peer ID in question has been received.

Please note that BGT uses UDP for all of its network communication. Thus, for a server to be able to accept incoming connections on a particular port, this port must have been properly opened to allow incoming UDP traffic.

Example 1 (server):
// Create a server that listens on port 10000, and sends back any messages it receives.

network  host;

void  main()
{
show_game_window("Game  Server");
tts_voice  voice;
if(host.setup_server(10000,  1,  100)==false)
{
voice.speak_wait("Error.  The  server  could  not  be  set  up.");
exit();

}
voice.speak("Welcome  to  the  echo  server.  This  server  will  accept  up  to  100  connections,  and  send  back  anything  it  receives.");
network_event  event;
while(true)
{
event=host.request();
if(event.type==event_connect)
{
voice.speak_interrupt("Peer  number  "  +  event.peer_id  +  "  connected  from  "  +  host.get_peer_address(event.peer_id)  +  ".");
voice.speak("Peers  now  connected:  "  +  host.connected_peers  +  ".");
}
if(event.type==event_receive)
{
host.send_unreliable(event.peer_id,  event.message,  event.channel);
}
if(event.type==event_disconnect)
{
voice.speak_interrupt("Peer  number  "  +  event.peer_id  +  "  just  disconnected.");
voice.speak("Peers  now  connected:  "  +  host.connected_peers  +  ".");
}
if(key_down(KEY_LMENU)  and  key_pressed(KEY_F4))
{
voice.speak_interrupt("Exiting.");
disconnect_everyone();
// Just in case the voice hasn't finished speaking when this function returns, we wait for it to finish before actually exiting.
while(voice.speaking==true)
{
wait(5);
}
exit();
}
wait(5);
}
}

void  disconnect_everyone()
{

/*
This function first gets a list of all connected peers and disconnects them. Then it waits until it has received the same number of disconnect notifications as it sent out disconnect requests, and then returns. This may take some time, however, so we set up a timeout after which we return anyway.

Note that we are not processing messages in this event loop. You will obviously want to do this if you have other connections to manage at the same time.
*/

timer  timeout;
uint[]  peer_list=host.get_peer_list();
int  expected_disconnects=peer_list.length();
for(uint  i=0;i  <  peer_list.length();i++)
{
host.disconnect_peer(peer_list[i]);
}
network_event  event;
while(expected_disconnects>0)
{
event=host.request();
if(event.type==event_disconnect)
{
expected_disconnects-=1;
}
if(timeout.elapsed>=1000)
{
break;
}
wait(5);
}
}

Example  2  (client):

// Create a client that connects to a server, and gathers some statistics about the connection.

network  host;

void  main()
{
show_game_window("Game  Client");
tts_voice  voice;
uint server_id=0;  // This is used to store the ID of the remote peer that will be our server.
if(host.setup_client(1,  1)==false)
{
voice.speak_wait("Error.  The  client  could  not  be  set  up.");
exit();
}
voice.speak("Client  started.");
voice.speak("Please  enter  the  IP  address  or  host  name  that  you  wish  to  connect  to.");
string  address=input_box("Address",  "Please  enter  the  host  name  or  address  to  connect  to.");
if(address=="")
{
voice.speak_interrupt_wait("Exiting.");
exit();
}
voice.speak_interrupt("Connecting,  please  wait.");
host.connect(address,  10000);
network_event  event;

// In this loop, we just wait for either a connect or disconnect event as this is always the first thing that we'll get.

while(true)
{
event=host.request();
if(event.type==event_connect)
{
voice.speak_interrupt("The  connection  to  "  +  host.get_peer_address(event.peer_id)  +  "  succeeded!");
server_id=event.peer_id;
break;
}
if(event.type==event_disconnect)
{
voice.speak_interrupt_wait("The  connection  to  "  +  host.get_peer_address(event.peer_id)  +  "  failed.  Exiting.");
exit();
}
if(key_down(KEY_LMENU)  and  key_pressed(KEY_F4))
{
host.destroy();
voice.speak_interrupt_wait("Exiting.");
exit();
}
wait(5);
}

/*
Since we got here, we know that we have an active connection and can begin gathering statistics. We do this by sending roughly 30 packets per second, for 10 seconds. Then, we disconnect and speak the statistics.
*/

timer  message_trigger;
timer  total_time;
int  sent_messages=0;
int  received_messages=0;
while(true)
{
event=host.request();
if(event.type==event_disconnect)
{
host.destroy();
voice.speak_interrupt_wait("The  server  died.  Exiting.");
exit();
}
if(event.type==event_receive)
{

received_messages+=1;
}
if(message_trigger.elapsed>=33)
{
message_trigger.restart();
host.send_unreliable(server_id,  "Hello!",  0);
sent_messages+=1;
}
if(total_time.elapsed>=10000)
{
// The statistics are gathered, so we speak them.
voice.speak_interrupt("Statistics:  Messages  sent,  "  +  sent_messages  +  ".  Messages  received,  "  +  received_messages  +  ".  Average  round  trip  time,  "  +  host.get_peer_average_round_trip_time(server_id)  +  "  milliseconds.  Exiting.");
disconnect_everyone();

/*
Just in case the voice hasn't finished speaking when this function returns, we wait for it to finish before actually exiting. We also allow the user to press alt+f4 since the message is pretty long.
*/
while(voice.speaking==true)
{
if(key_down(KEY_LMENU)  and  key_pressed(KEY_F4))
{
break;
}
wait(5);
}
exit();
}
if(key_down(KEY_LMENU)  and  key_pressed(KEY_F4))
{
voice.speak_interrupt("Exiting.");
disconnect_everyone();
// Just in case the voice hasn't finished speaking when this function returns, we wait for it to finish before actually exiting.
while(voice.speaking==true)
{
wait(5);
}
exit();
}
wait(5);
}
}

void  disconnect_everyone()
{

/*
This is the exact same function as the one found in the server, as it works equally well for both.

This function first gets a list of all connected peers and disconnects them. Then it waits until it has received the same number of disconnect notifications as it sent out disconnect requests, and then returns. This may take some time, however, so we set up a timeout after which we return anyway.

Note that we are not processing messages in this event loop. You will obviously want to do this if you have other connections to manage at the same time.
*/

timer  timeout;
uint[]  peer_list=host.get_peer_list();
int  expected_disconnects=peer_list.length();
for(uint  i=0;i  <  peer_list.length();i++)
{
host.disconnect_peer(peer_list[i]);
}
network_event  event;
while(expected_disconnects>0)
{
event=host.request();
if(event.type==event_disconnect)
{
expected_disconnects-=1;
}
if(timeout.elapsed>=1000)
{

break;
}
wait(5);
}
}

network object

The network object is used to allow games to be played online with other live players in a peer to peer environment.

network()

Parameters:
None.

Remarks:
This object allows the creation of multiplayer games of any complexity. One player will usually act as a server while the rest connect to that server.

The procedure when writing a networked game in BGT is to either set up a server and then wait for incoming connections, or to connect to a remote computer using its IP address and a designated port. Port numbers range from 1 to 65535, however most ports below 1000 are reserved and should thus be avoided. A server must be listening on the same port as the client uses when connecting
to the server, in order for a connection to be successfully established.

There are two components to the networking system in BGT. There is the network object itself, and an object called network_event, which contains information about a received event. This network event is filled in with information when a call is made to the request method, and you then examine the properties of this returned object in order to determine what has occured.

As soon as you start using the network object, you must continuously call the request method in order to get events from the outside world. A server will receive an event when a new person has connected, for instance, and a client will receive an event when an attempted connection to a server has either succeeded or failed.

Along with each event, you will get a so called peer ID. A peer ID is a unique number that identifies a peer. When a client attempts to connect to a server, for example, the connect method will return a peer ID immediately. This peer ID will then be in the process of connecting, and will not be active until a connect event has been received which contains the same ID. Similarly, when a server
receives a new incoming connection event, this event will always contain a peer ID so that the server is able to uniquely identify this peer when sending data to and receiving data from it.

Once a connection has been successfully established between two or more players, these players will begin passing data back and forth between one another by sending so called packets. A packet is basically a sequence of bytes given as a string. The packets will always arrive in the same order in which they were sent, meaning that you will never get an older packet after a newer one has
already been delivered. You can send packets in one of two ways, reliably and unreliably.

Unreliable packets are by far the quickest way of getting data across to another person. An unreliable packet will be sent at the maximum speed provided by the connection, but is not guaranteed to arrive at the destination. It is, however, guaranteed to be properly sequenced if it does arrive. Unreliable packets are useful when you have data that needs to be sent out very frequently, but you only
really care about the most recent data. You are not particularly interested in another player's position from 2 seconds ago, for instance, but rather you want to know where they are at this very moment.

Reliable packets are always guaranteed to get through, and always in the proper order just like unreliable ones. This reliability comes at a cost, however. Since the internet is unreliable by nature, BGT needs to wait for an acknowledgement from the other person to say that the packet was in fact received. If no such acknowledgement is received within a reasonable amount of time, a new attempt
to send the packet is made. The same procedure is repeated over and over again until an acknowledgement is received, or until enough time has elapsed to deem the connection as dead. This may cause considerable delays in the transmition as compared to unreliable packets, and should thus only be used for things where speed is not critical.

A problem arises when trying to send both unreliable and reliable packets at the same time. Since packets are required to be sequenced and reliable packets require verification before they are delivered, they will stall any unreliable packets that come after it in the sequence. Then, once the reliable packet is received the unreliable ones will arrive all at once. To prevent this, BGT uses channels. A
channel is an independently sequenced queue of packets, meaning that you can send both reliable and unreliable packets at the same time on different channels without them blocking one another. Channels also allow you to structure your data more efficiently, by for example sending position updates on one channel and shooting events on another.

Care should be taken when using multiple channels, to avoid a situation where one channel lags behind the others in time due to multiple reliable packets being stalled for example. Thus, if sequencing is important for a series of events, it may be necessary to put them on the same channel and wait for them to actually get through. Different channels are only beneficial when the usefulness of one
packet is not strictly dependent on another packet having been delivered before it.

Channels are 0 based, meaning that if you request a connection with 3 channels you may use channels 0, 1, and 2 when sending your data.

When disconnecting from a remote peer, you have three methods to choose from. disconnect_peer_forcefully, disconnect_peer, and disconnect_peer_softly. More information about how each method works can be found in that method's chapter, but the choice of method all has to do with how and when you want the remote peer to be notified of your disconnection. After calling
disconnect_peer or disconnect_peer_softly, you must keep checking for events on the host as the disconnection will not be complete until a disconnect event containing the peer ID in question has been received.

Please note that BGT uses UDP for all of its network communication. Thus, for a server to be able to accept incoming connections on a particular port, this port must have been properly opened to allow incoming UDP traffic.

Example 1 (server):
// Create a server that listens on port 10000, and sends back any messages it receives.

network  host;

void  main()
{
show_game_window("Game  Server");
tts_voice  voice;
if(host.setup_server(10000,  1,  100)==false)
{
voice.speak_wait("Error.  The  server  could  not  be  set  up.");
exit();

}
voice.speak("Welcome  to  the  echo  server.  This  server  will  accept  up  to  100  connections,  and  send  back  anything  it  receives.");
network_event  event;
while(true)
{
event=host.request();
if(event.type==event_connect)
{
voice.speak_interrupt("Peer  number  "  +  event.peer_id  +  "  connected  from  "  +  host.get_peer_address(event.peer_id)  +  ".");
voice.speak("Peers  now  connected:  "  +  host.connected_peers  +  ".");
}
if(event.type==event_receive)
{
host.send_unreliable(event.peer_id,  event.message,  event.channel);
}
if(event.type==event_disconnect)
{
voice.speak_interrupt("Peer  number  "  +  event.peer_id  +  "  just  disconnected.");
voice.speak("Peers  now  connected:  "  +  host.connected_peers  +  ".");
}
if(key_down(KEY_LMENU)  and  key_pressed(KEY_F4))
{
voice.speak_interrupt("Exiting.");
disconnect_everyone();
// Just in case the voice hasn't finished speaking when this function returns, we wait for it to finish before actually exiting.
while(voice.speaking==true)
{
wait(5);
}
exit();
}
wait(5);
}
}

void  disconnect_everyone()
{

/*
This function first gets a list of all connected peers and disconnects them. Then it waits until it has received the same number of disconnect notifications as it sent out disconnect requests, and then returns. This may take some time, however, so we set up a timeout after which we return anyway.

Note that we are not processing messages in this event loop. You will obviously want to do this if you have other connections to manage at the same time.
*/

timer  timeout;
uint[]  peer_list=host.get_peer_list();
int  expected_disconnects=peer_list.length();
for(uint  i=0;i  <  peer_list.length();i++)
{
host.disconnect_peer(peer_list[i]);
}
network_event  event;
while(expected_disconnects>0)
{
event=host.request();
if(event.type==event_disconnect)
{
expected_disconnects-=1;
}
if(timeout.elapsed>=1000)
{
break;
}
wait(5);
}
}

Example  2  (client):

// Create a client that connects to a server, and gathers some statistics about the connection.

network  host;

void  main()
{
show_game_window("Game  Client");
tts_voice  voice;
uint server_id=0;  // This is used to store the ID of the remote peer that will be our server.
if(host.setup_client(1,  1)==false)
{
voice.speak_wait("Error.  The  client  could  not  be  set  up.");
exit();
}
voice.speak("Client  started.");
voice.speak("Please  enter  the  IP  address  or  host  name  that  you  wish  to  connect  to.");
string  address=input_box("Address",  "Please  enter  the  host  name  or  address  to  connect  to.");
if(address=="")
{
voice.speak_interrupt_wait("Exiting.");
exit();
}
voice.speak_interrupt("Connecting,  please  wait.");
host.connect(address,  10000);
network_event  event;

// In this loop, we just wait for either a connect or disconnect event as this is always the first thing that we'll get.

while(true)
{
event=host.request();
if(event.type==event_connect)
{
voice.speak_interrupt("The  connection  to  "  +  host.get_peer_address(event.peer_id)  +  "  succeeded!");
server_id=event.peer_id;
break;
}
if(event.type==event_disconnect)
{
voice.speak_interrupt_wait("The  connection  to  "  +  host.get_peer_address(event.peer_id)  +  "  failed.  Exiting.");
exit();
}
if(key_down(KEY_LMENU)  and  key_pressed(KEY_F4))
{
host.destroy();
voice.speak_interrupt_wait("Exiting.");
exit();
}
wait(5);
}

/*
Since we got here, we know that we have an active connection and can begin gathering statistics. We do this by sending roughly 30 packets per second, for 10 seconds. Then, we disconnect and speak the statistics.
*/

timer  message_trigger;
timer  total_time;
int  sent_messages=0;
int  received_messages=0;
while(true)
{
event=host.request();
if(event.type==event_disconnect)
{
host.destroy();
voice.speak_interrupt_wait("The  server  died.  Exiting.");
exit();
}
if(event.type==event_receive)
{

received_messages+=1;
}
if(message_trigger.elapsed>=33)
{
message_trigger.restart();
host.send_unreliable(server_id,  "Hello!",  0);
sent_messages+=1;
}
if(total_time.elapsed>=10000)
{
// The statistics are gathered, so we speak them.
voice.speak_interrupt("Statistics:  Messages  sent,  "  +  sent_messages  +  ".  Messages  received,  "  +  received_messages  +  ".  Average  round  trip  time,  "  +  host.get_peer_average_round_trip_time(server_id)  +  "  milliseconds.  Exiting.");
disconnect_everyone();

/*
Just in case the voice hasn't finished speaking when this function returns, we wait for it to finish before actually exiting. We also allow the user to press alt+f4 since the message is pretty long.
*/
while(voice.speaking==true)
{
if(key_down(KEY_LMENU)  and  key_pressed(KEY_F4))
{
break;
}
wait(5);
}
exit();
}
if(key_down(KEY_LMENU)  and  key_pressed(KEY_F4))
{
voice.speak_interrupt("Exiting.");
disconnect_everyone();
// Just in case the voice hasn't finished speaking when this function returns, we wait for it to finish before actually exiting.
while(voice.speaking==true)
{
wait(5);
}
exit();
}
wait(5);
}
}

void  disconnect_everyone()
{

/*
This is the exact same function as the one found in the server, as it works equally well for both.

This function first gets a list of all connected peers and disconnects them. Then it waits until it has received the same number of disconnect notifications as it sent out disconnect requests, and then returns. This may take some time, however, so we set up a timeout after which we return anyway.

Note that we are not processing messages in this event loop. You will obviously want to do this if you have other connections to manage at the same time.
*/

timer  timeout;
uint[]  peer_list=host.get_peer_list();
int  expected_disconnects=peer_list.length();
for(uint  i=0;i  <  peer_list.length();i++)
{
host.disconnect_peer(peer_list[i]);
}
network_event  event;
while(expected_disconnects>0)
{
event=host.request();
if(event.type==event_disconnect)
{
expected_disconnects-=1;
}
if(timeout.elapsed>=1000)
{

break;
}
wait(5);
}
}

network_event object

The network_event object is part of the network object and is used to retrieve information about a received event.

network_event()

Parameters:
None.

Remarks:
This object is used as a data type containing read-only properties about a received event, as returned by the request method of
the network object.

Example:
See the main network chapter.

network_event object

The network_event object is part of the network object and is used to retrieve information about a received event.

network_event()

Parameters:
None.

Remarks:
This object is used as a data type containing read-only properties about a received event, as returned by the request method of
the network object.

Example:
See the main network chapter.

network_event object

int type
Determines the type of event that is being received. This can be any of the following values:

EVENT_NONE (0) - No event is being received at that time.
EVENT_CONNECT (1) - A peer has just connected.
EVENT_RECEIVE (2) - A packet has been received from a peer.
EVENT_DISCONNECT (3) - A remote peer has just disconnected.

The default value for this property is event_none. This property cannot be modified from the script.

uint peer_id
The unique ID of the peer to whom the event relates. This is valid for all events except EVENT_NONE, and is set to 0
otherwise. This property cannot be modified from the script.

int channel
The channel on which the event was received. This is only valid for events of type EVENT_RECEIVE, and is set to 0 otherwise.

string message
The message that was received. This is only valid for events of type EVENT_RECEIVE, and is an empty string otherwise. This
property cannot be modified from the script.

network object

This method will attempt to establish a connection between yourself and a remote peer.

uint connect(string host, int port)

Parameters:
host
The host name or IP address to connect to.
port
The port that the other person is listening on.

Return value:
A unique peer ID that will be used when sending data on success, or 0 on error.

Remarks:
When this method returns a peer ID, this peer is not yet in a connected state. The connection will take some time to perform, and
you thus have to call the request method in order to retrieve events from the host. Eventually you will receive either an event of
type connect or of type disconnect, that contains the same peer ID as that returned by the call to connect. If a connect event is
received, you know that the connection attempt succeeded and you may begin using the peer ID to send messages to this peer.
If a disconnect event is fired, however, the connection failed and the peer ID should be discarded.

Example:
See the main network chapter.

network object

This method destroys a host.

bool destroy()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method will return the host to its initial uninitialized state, and destroy all its peers. Before it is destroyed, the host calls
disconnect_peer_forcefully on all the connected peers, in order to inform them of its action in the majority of cases so that they
do not simply time out. Since this does not guarantee that the remote peers are notified, it may not be sufficient and you will want
to disconnect every peer manually before destroying the host if you are concerned about the possibility of some of the remote
peers having to time out.

Example:
See the main network chapter.

network object

This method will disconnect a connected peer as soon as possible, while still giving them a guaranteed notification.

bool disconnect_peer(uint peer_id)

Parameters:
peer_id
The unique ID of the peer to disconnect.

Return value:
true on success, false on failure.

Remarks:
This function will send out a disconnection request immediately, without first taking the time to send any packets that may
previously have been queued. This means that the disconnection will happen as soon as possible, while still giving the remote
peer a guaranteed notification.

When this function returns, the disconnection is not yet complete. Only when an event of type disconnect that contains the peer
ID in question has been received, is the disconnection successful.

Example:
See the main network chapter.

network object

This method will disconnect a connected peer immediately.

bool disconnect_peer_forcefully(uint peer_id)

Parameters:
peer_id
The unique ID of the peer to disconnect.

Return value:
true on success, false on failure.

Remarks:
This method will disconnect a peer immediately, without doing any kind of verification with the remote peer. No disconnect event
is generated after this call, instead an unreliable disconnection notification is sent out and the peer is then reset immediately. Since
the notification is not guaranteed to arrive, it may be that the remote peer times out on its own eventually. Thus, this method
should only be used when you do not have the time to wait for the proper disconnection procedure to be performed.

Example:
See the main network chapter.

network object

This method will disconnect a connected peer once all previously queued packets have been sent out, giving them a guaranteed
notification in the end.

bool disconnect_peer_softly(uint peer_id)

Parameters:
peer_id
The unique ID of the peer to disconnect.

Return value:
true on success, false on failure.

Remarks:
This function will send out a disconnection request, after first having taken the time to send any packets that may previously have
been queued. This means that the disconnection may happen some time in the future and that other packets may be received by
the remote peer before the disconnection notification, although this notification is still guaranteed to arrive in the end.

When this function returns, the disconnection is not yet complete. Only when an event of type disconnect that contains the peer
ID in question has been received, is the disconnection successful.

Example:
See the main network chapter.

network object

This method will retrieve the host or IP address for a given peer.

string get_peer_address(uint peer_id)

Parameters:
peer_id
The unique ID of the peer to retrieve the IP address for.

Return value:
The IP address on success, or a blank string on failure.

Remarks:
This function is useful when you want to display the IP address of a person who just connected to a listening server, for instance.

Example:
See the main network chapter.

network object

This method retrieves the average round trip time between yourself and a connected peer.

double get_peer_average_round_trip_time(uint peer_id)

Parameters:
peer_id
The peer ID to retrieve the average round trip time for.

Return value:
The average round trip time between you and the specified peer in milliseconds on success, or -1 on error.

Remarks:
The round trip time is the time that it takes for a packet to travel from your location to the remote peer, and back to you again.
Thus, if you have an average round trip time of 100 milliseconds it means that it takes roughly 50 milliseconds for one of your
packets to arrive at the remote peer's location and vise versa.

The round trip time is calculated when reliable packets are sent out and subsequent acknowledgements of these are received.
When a connection is first established the average round trip time will thus not be accurate, but will stabilise over time. The rate
of this is determined by how many reliable packets that are sent out. BGT itself sends out reliable packets at regular intervals in
order to monitor the connection, which means that the average round trip time will eventually be accurate even if you don't send
out any reliable packets yourself, although this will take longer. Thus, don't rely on the average round trip time in the very
beginning of a connection.

Example:
See the main network chapter.

network object

This method will return an array containing the unique ID's of all connected peers.

uint[] get_peer_list()

Parameters:
None.

Return value:
An array containing the unique ID's of all connected peers on success, or an array with 0 elements on error.

Remarks:
This method is useful for sending messages to all connected peers, as the returned array can easily be used in a loop to send the
same message to everyone. Please note that the same functionality can be achieved by passing 0 as the peer ID to either the
send_unreliable or send_reliable methods.

This function is also useful when you wish to disconnect all peers gracefully before destroying the host.

Example:
See the main network chapter.

network object

This method is used to request events from the outside world.

network_event request()

Parameters:
None.

Return value:
Returns a network_event object containing all the information about the event received.

Remarks:
This method is used to poll the network for any events that are received from the outside, so that they can be properly handled
by your application.

To access the data that the request method returns, simply access the properties of the network_event object as you would any
other object properties. If no event was found or if an error occured, a network_event object with the type property set to
event_none and all the other properties set to their default values is returned.

You must be sure to call this method continuously in your game as soon as you begin dealing with one or more remote peers.
Failing to do so will cause the communication to lag.

Example:
See the main network chapter.

network object

This method will send a reliable packet to a given peer.

bool send_reliable(uint peer_id, string packet, int channel)

Parameters:
peer_id
The unique ID of the peer that is to receive the message.
packet
The message to be sent.
channel
The channel that the message should be sent on.

Return value:
true on success, false on failure.

Remarks:
This method sends a reliable packet to a given peer, or to all connected peers if the peer ID is set to 0.

Reliable packets are always guaranteed to get through, and always in the proper order just like unreliable ones. This reliability
comes at a cost, however. Since the internet is unreliable by nature, BGT needs to wait for an acknowledgement from the other
person to say that the packet was in fact received. If no such acknowledgement is received within a reasonable amount of time, a
new attempt to send the packet is made. The same procedure is repeated over and over again until an acknowledgement is
received, or until enough time has elapsed to deem the connection as dead. This may cause considerable delays in the transmition
as compared to unreliable packets, and should thus only be used for things where speed is not critical.

For more information about the differences between unreliable and reliable packets, as well as information about how to
sequence packets using channels, see the main network chapter.

Example:
See the main network chapter.

network object

This method will send an unreliable packet to a given peer.

bool send_unreliable(uint peer_id, string packet, int channel)

Parameters:
peer_id
The unique ID of the peer that is to receive the message.
packet
The message to be sent.
channel
The channel that the message should be sent on.

Return value:
true on success, false on failure.

Remarks:
This method sends an unreliable packet to a given peer, or to all connected peers if the peer ID is set to 0.

Unreliable packets are by far the quickest way of getting data across to another person. An unreliable packet will be sent at the
maximum speed provided by the connection, but is not guaranteed to arrive at the destination. It is, however, guaranteed to be
properly sequenced if it does arrive. Unreliable packets are useful when you have data that needs to be sent out very frequently,
but you only really care about the most recent data. You are not particularly interested in another player's position from 2
seconds ago, for instance, but rather you want to know where they are at this very moment.

For more information about the differences between unreliable and reliable packets, as well as information about how to
sequence packets using channels, see the main network chapter.

Example:
See the main network chapter.

network object

This method will limit the incoming and outgoing bandwidth of the host.

bool set_bandwidth_limits(double incoming_bandwidth, double outgoing_bandwidth)

Parameters:
incoming_bandwidth
The number of incoming bytes to allow per second.
outgoing_bandwidth
The number of outgoing bytes to allow per second.

Return value:
true on success, false on failure.

Remarks:
BGT will continuously monitor the connection and try to detect bandwidth overflows dynamically, but this method gives fine
control over exactly how much data that is sent and received to and from all peers every second. Setting the outgoing bandwidth
to 4096, for instance, will cause BGT to stall delivery of reliable packets and strategically drop unreliable ones in order to try to
prevent the outgoing bandwidth from ever exceeding 4 kb per second.

Setting either of these parameters to 0 will cause BGT to rely entirely upon its dynamic bandwidth throttling, which is the default
behavior.

Example:
See the main network chapter.

network object

This method will initialise and set up the network object in client mode.

bool setup_client(int channels, int max_peers)

Parameters:
channels
The number of channels that are used by the client. The maximum is 100.
max_peers
The maximum number of peers that the client can be connected with simultaneously. The maximum is 4000.

Return value:
true if the client was set up successfully, false otherwise.

Remarks:
This method is used to set up the network object in client mode. This means that it will not listen for any incoming connections.

Please note that it is perfectly legal for a client to make more than one outgoing connection at once, hence the max_peers
parameter.

Example:
See the main network chapter.

network object

This method will initialise and set up the network object in server mode.

bool setup_server(int listening_port, int channels, int max_peers)

Parameters:
listening_port
The port that is to be used while listening for a connection.
channels
The number of channels that are used by the client. The maximum is 100.
max_peers
The maximum number of peers that the server can be connected with simultaneously. The maximum is 4000.

Return value:
true if the server was set up successfully, false otherwise.

Remarks:
This method is used to set up the network object in server mode. Once the server has been set up, it will automatically listen for
connections on the given port. Whenever a new peer tries to connect, an event of type connect will be received.

Please note that it is perfectly legal for a server to make outgoing connections as well. These outgoing connections will be part of
the same list of peers as the incoming connections, meaning that both incoming and outgoing connections on the server will count
towards the maximum peer count. Thus, a server may act both as a server and as a client at one and the same time.

Example:
See the main network chapter.

network object

int connected_peers
Specifies how many peers are connected to the current session, or -1 on error. This property cannot be modified from the script.

double bytes_sent
Specifies the number of bytes that have been sent in total throughout the current session to all peers, including packet headers
and other data not actually part of user packets, or -1 on error. This property cannot be modified from the script.

double bytes_received
Specifies the number of bytes received throughout the current session from all peers, including packet headers and other data not
actually part of user packets, or -1 on error. This property cannot be modified from the script.

bool active
Returns true while the host is active (e.g. between calls to one of the setup methods and the destroy method), or false otherwise.
This property cannot be modified from the script.

pack_file object

The pack_file object is used to package several files into one.

pack_file()

Parameters:
None.

Remarks:
A pack file is generally useful if you have a lot of files that you wish to distribute as one file on disk. This most commonly applies
to sounds, but can include other data such as game level layouts etc.

When a pack_file object is first created, it will not be active. To activate it you must call either the "create" or the "open" method
on the object, specifying a file that should be associated with it. You may call these methods again at any time if you wish to
associate another pack with the object, at which point the old association (if any) will be cleared. This, in short, allows you to
reuse the same pack_file object for manipulating more than one pack.

Please note that storing files inside a pack in no way enhances security. If you wish to protect your files from being tampered with
or read by others, use the encryption functions on them before adding them to the pack.

Example:
// Create a pack file and add some test files.

void  main()
{
pack_file  test;
test.create("pack.dat");
test.add_file("test1.txt","t1");
test.add_file("test2.txt","t2");
test.add_file("test3.txt","t3");
test.close();
}

pack_file object

The pack_file object is used to package several files into one.

pack_file()

Parameters:
None.

Remarks:
A pack file is generally useful if you have a lot of files that you wish to distribute as one file on disk. This most commonly applies
to sounds, but can include other data such as game level layouts etc.

When a pack_file object is first created, it will not be active. To activate it you must call either the "create" or the "open" method
on the object, specifying a file that should be associated with it. You may call these methods again at any time if you wish to
associate another pack with the object, at which point the old association (if any) will be cleared. This, in short, allows you to
reuse the same pack_file object for manipulating more than one pack.

Please note that storing files inside a pack in no way enhances security. If you wish to protect your files from being tampered with
or read by others, use the encryption functions on them before adding them to the pack.

Example:
// Create a pack file and add some test files.

void  main()
{
pack_file  test;
test.create("pack.dat");
test.add_file("test1.txt","t1");
test.add_file("test2.txt","t2");
test.add_file("test3.txt","t3");
test.close();
}

pack_file object

This method will add a file to the currently open pack file.

bool add_file(string file_on_disk, string internal_name)

Parameters:
file_on_disk
The filename stored on disk to add to the pack file.
internal_name
An internal storage name that should be used to refer to it inside the pack file.

Return value:
true on success, false on failure.

Remarks:
The pack file must be opened in write mode with the create method for this operation to be successful.

The internal_name parameter is simply used to refer to the file inside the pack. This can contain any characters and can be laid
out in any format.

Example:
// Create a pack file and add some test files.

void  main()
{
pack_file  test;
test.create("pack.dat");
test.add_file("test1.txt","t1");
test.add_file("test2.txt","t2");
test.add_file("test3.txt","t3");
test.close();
}

pack_file object

This method will close any pack file associated with the object.

bool close()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method is executed automatically when the object is destroyed, so you need not worry about calling it to clean up after
yourself at the end of your script for example. This method is also executed if you call either the create or open method on a
pack_file object that already has a pack associated with it.

Example:
// Create a pack file and add some test files.

void  main()
{
pack_file  test;
test.create("pack.dat");
test.add_file("test1.txt","t1");
test.add_file("test2.txt","t2");
test.add_file("test3.txt","t3");
test.close();
}

pack_file object

This method will create a pack file and open it in write mode.

bool create(string filename)

Parameters:
filename
The file to create.

Return value:
true on success, false on failure.

Remarks:
If the file already exists, it will be overwritten.

Example:
// Create a pack file and add some test files.

void  main()
{
pack_file  test;
test.create("pack.dat");
test.add_file("test1.txt","t1");
test.add_file("test2.txt","t2");
test.add_file("test3.txt","t3");
test.close();
}

pack_file object

This method will extract a file from an open pack file back onto disk.

bool extract_file(string internal_name, string file_on_disk)

Parameters:
internal_name
The internal storage name of a file inside the pack.
file_on_disk
The filename to use when extracting back onto disk.

Return value:
true on success, false on failure.

Remarks:
This method will work on a pack file opened for reading.

Example:
// Open a pack file and extract some files.

void  main()
{
pack_file  test;
test.open("pack.dat");
test.extract_file("t1","test1.txt");
test.extract_file("t2","test2.txt");
test.extract_file("t3","test3.txt");
test.close();
}

pack_file object

This method checks whether a given file exists in the pack.

bool file_exists(string internal_name)

Parameters:
internal_name
The name of the file to search for in the pack.

Return value:
true if the file exists, false otherwise.

Remarks:
This method will work in a pack file opened for reading.

Example:
// Check if a file exists.

void  main()
{
pack_file  test;
test.open("pack.dat");
if(test.file_exists("t1");
{
alert("Information",  "The  file  t1  exists  in  the  pack.dat  file.");
}
else
{
alert("Oops",  "T1  does  not  exist  in  the  pack.dat  file.");
}
}

pack_file object

This method will pass a file within a pack to a file object for reading.

file@ get_file(string internal_name)

Parameters:
internal_name
The name of the file to pass to a file object.

Return value:
A file object pointing to the data on success, or an inactive file object on failure.

Remarks:
This method passes data from a file inside a pack to a standard file object. This allows a file to be read in chunks from inside the
pack without the need to extract it to disk. The file object does not contain a copy of the data, it reads from the original pack but
only the relevant section of the file, so that you need not worry about accessing other data in the pack by accident.

Remember to always check the active property of the file object after this call, in order to determine whether or not the retrieval
was successful.

Example:
// Read a file from inside a pack.

void  main()
{
file@  data;
pack_file  pack;
pack.open("pack.dat");
@data=pack.get_file("t1");
if(data.active==false)
{
alert("Oops","The  file  could  not  be  read  from  the  pack");
pack.close();
}
else
{
string  text=data.read(0);
data.close();
pack.close();
alert("information",  "Data  in  the  t1  file  in  pack.dat  reads  the  following:\r\n"+text);
}
}

pack_file object

This method will list all the files in the pack.

string[] list_files()

Parameters:
None.

Return value:
An array of files on success, or an empty array on failure.

Remarks:
Although internal names can take the form of directory formatting, the pack format cannot differentiate between directories and
files. Therefore if you have an internal name of music/level1.ogg, that is what will appear in the relevant entry in the list.

Files will be listed in the order they were added to the pack file.

This method can only be used on a pack opened for reading.

Example:
// List the files in a pack.

void  main()
{
pack_file  test;
test.open("pack.dat");
string[]  list;
list=test.list_files();
for(uint  counter=0;  counter<list.length();  counter++)
{
alert("File  "+counter,list[counter]);
}
}

pack_file object

This method will open an existing pack file for reading.

bool open(string filename)

Parameters:
filename
The pack file on disk to open.

Return value:
true on success, false on failure.

Remarks:
When you call this method, the pack file will be opened in read mode. Please note that a pack file cannot be added to once
closed unless it is recreated from scratch.

Note that both \ and / can be used to specify paths.

If you pass * to this function, the engine will attempt to open a pack file that has been included into the program itself. For more
information on how this is done, see the tutorial on packaging files with your executable.

Example:
// Open a pack file and extract some data.

void  main()
{
pack_file  test;
test.open("pack.dat");
test.extract_file("t1","test1.txt");
test.extract_file("t2","test2.txt");
test.extract_file("t3","test3.txt");
test.close();
}

pack_file object

double file_count
Returns the number of files stored in the currently open pack, or -1 on error. This property cannot be modified from the script.

bool active
This is set to true when a pack file is associated with the object, and to false otherwise. This property cannot be modified from
the script.

pathfinder object

The pathfinder object is used to find a path between two locations.

pathfinder()

Parameters:
None.

Remarks:
Pathfinding is widely used in artificial intelligence, to make computer entities appear to make smart and calculated decisions in an arbitrarily complex environment.

The environment is represented as a map, which is a two dimensional grid. When finding a path from one location on the map to another, the pathfinder investigates numerous squares on the map when determining the best path. For each square that it investigates it calls a function in your script known as the pathfinder callback. This function is meant to tell the pathfinder the risk, or cost, of moving to this particular square. For instance, a square with grass is a lot less dangerous than a square with fire. By use of this callback you may impose any
types of rules you wish, such as deciding that certain creatures in your game are allowed to walk through fire while others are not. You would determine what type of creature you were dealing with inside the callback, and then return an appropriate risk factor for the fire square.

For more detailed information about how pathfinding works in practise, see the pathfinding tutorial.

Example:
// Let the user walk around on a grid and place out walls, as well as a starting square.
//  Pressing  enter  will  generate  a  path  between  the  starting  square  and  the  user's  current  location.
// All the keyboard commands are spoken at the start of the program.

const  int  maze_size=10;
// The maze is always a square, so setting this constant to 10 generates a 10 by 10 maze (100 squares).

//  Declare  the  two  dimensional  maze  array.
int[][]  maze;

//  Now  declare  the  variables  that  will  hold  the  user's  current  coordinates,  and  the  coordinates  of  the  starting  square.
int  x=0;
int  y=0;
int  start_x=-1;
int  start_y=-1;

//  Finally,  declare  the  pathfinder  and  tts_voice  objects.
pathfinder  holmes;
tts_voice  speech;

void  main()
{
show_game_window("Maze  Builder");

// Resize the grid to the proper dimentions and fill it with empty space.
maze.resize(maze_size);
for(int  i=0;i<maze_size;i++)
{
maze[i].resize(maze_size);
for(int  i2=0;i2<maze_size;i2++)
maze[i][i2]=0;
}

// Set up the pathfinder.
if(holmes.create_map(maze_size,  maze_size)==false)
{
speech.speak_wait("Could  not  set  up  the  map.");
exit();
}
holmes.set_callback_function(maze_callback);

speech.speak("Use the four arrow keys to navigate on the grid, and press escape to quit. Press space to place out a wall, and delete to remove one. Press s to set the starting square, and enter to ask the computer to walk from the starting square to your current location. Press the letter keys x and y to check your current coordinates. Press the d key to toggle whether diagonal paths are allowed or not.");

// Run the main game loop, which exits when the user presses escape.
while(key_down(KEY_ESCAPE)==false)
{

// Allow the user to move, but not outside the grid boundries.
if(key_pressed(KEY_LEFT))
{
x-=1;
if(x<0)
x+=1;
else
speak_location();
}
if(key_pressed(KEY_RIGHT))
{
x+=1;
if(x>=maze_size)
x-=1;
else
speak_location();
}
if(key_pressed(KEY_DOWN))
{
y-=1;
if(y<0)
y+=1;
else
speak_location();
}
if(key_pressed(KEY_UP))
{
y+=1;
if(y>=maze_size)
y-=1;
else
speak_location();
}

// Place out and remove walls.
if(key_pressed(KEY_SPACE))
{
if(maze[x][y]==1)
{
speech.speak_interrupt("Already  a  wall  there.");
}
else
{
maze[x][y]=1;
speech.speak_interrupt("Added  wall.");
}
}
if(key_pressed(KEY_DELETE))
{
if(maze[x][y]==0)
{
speech.speak_interrupt("No  wall  there.");
}
else
{
maze[x][y]=0;
speech.speak_interrupt("Removed  wall.");
}
}

// Find a path from the starting square to the user's current location.
if(key_pressed(KEY_RETURN))
{
if(start_x==-1  or  start_y==-1)
{
speech.speak_interrupt("Please  specify  a  starting  square  by  pressing  s  somewhere.");
continue;
}
vector[]  path=holmes.find(start_x,  start_y,  x,  y,  "");

if(path.length()==0)
{
speech.speak_interrupt("No  path  found.");
continue;
}

// We now have a path, so we assemble a string with the information that should be spoken.
string  output="";

// We now go through the path and convert the absolute coordinates to relative moves like west, north, etc.
int  last_x=start_x;
int  last_y=start_y;
for(int  i=0;i<path.length();i++)
{

//  Be  nice  to  speech  synthesizers.
if(output!="")
{
output+=",  ";
}

//  Check  for  the  8  possible  directions,  inserting  messages  in  the  string  accordingly.
// Note that only four of these will actually be present in the path unless diagonal paths are enabled.
if(path[i].x<last_x  and  path[i].y==last_y)
output+="west";
if(path[i].x>last_x  and  path[i].y==last_y)
output+="east";
if(path[i].y<last_y  and  path[i].x==last_x)
output+="south";
if(path[i].y<last_y  and  path[i].x<last_x)
output+="south  west";
if(path[i].y<last_y  and  path[i].x>last_x)
output+="south  east";
if(path[i].y>last_y  and  path[i].x==last_x)
output+="north";
if(path[i].y>last_y  and  path[i].x<last_x)
output+="north  west";
if(path[i].y>last_y  and  path[i].x>last_x)
output+="north  east";

last_x=path[i].x;
last_y=path[i].y;
}
speech.speak_interrupt(output);
}

//  Set  the  starting  location.
if(key_pressed(KEY_S))
{
start_x=x;
start_y=y;
speech.speak_interrupt("Starting square set to x " + x + ", y " + y + ".");
}

//  Should  we  allow  diagonal  paths?
if(key_pressed(KEY_D))
{
if(holmes.allow_diagonals==true)
{
holmes.allow_diagonals=false;
speech.speak_interrupt("Diagonal  paths  are  now  forbidden.");
}
else
{
holmes.allow_diagonals=true;
speech.speak_interrupt("Diagonal  paths  are  now  allowed.");
}
}

//  Check  our  coordinates.

if(key_pressed(KEY_X))
{
speech.speak_interrupt("X  "  +  x);
}
if(key_pressed(KEY_Y))
{
speech.speak_interrupt("Y  "  +  y);
}
wait(5);
}

speech.speak_interrupt_wait("Thanks  for,  playing?  I  guess?");
}

// This function tells the user whether the square is free or has a wall on it, and whether it is the starting square.
void  speak_location()
{
if(maze[x][y]==1)
{
if(x==start_x  and  y==start_y)
speech.speak_interrupt("Starting  square  and  wall.");
else
speech.speak_interrupt("Wall.");
return;
}
if(x==start_x  and  y==start_y)
speech.speak_interrupt("Starting  square  and  floor.");
else
speech.speak_interrupt("Floor.");
}

// This is the pathfinder callback. We simply check if there is a wall on the square and if there is, return 10. If not, return 0.
int  maze_callback(int  x,  int  y,  int  parent_x,  int  parent_y,  string  user_data)
{
if(maze[x][y]==1)
{
return  10;
}
return  0;
}

pathfinder object

The pathfinder object is used to find a path between two locations.

pathfinder()

Parameters:
None.

Remarks:
Pathfinding is widely used in artificial intelligence, to make computer entities appear to make smart and calculated decisions in an arbitrarily complex environment.

The environment is represented as a map, which is a two dimensional grid. When finding a path from one location on the map to another, the pathfinder investigates numerous squares on the map when determining the best path. For each square that it investigates it calls a function in your script known as the pathfinder callback. This function is meant to tell the pathfinder the risk, or cost, of moving to this particular square. For instance, a square with grass is a lot less dangerous than a square with fire. By use of this callback you may impose any
types of rules you wish, such as deciding that certain creatures in your game are allowed to walk through fire while others are not. You would determine what type of creature you were dealing with inside the callback, and then return an appropriate risk factor for the fire square.

For more detailed information about how pathfinding works in practise, see the pathfinding tutorial.

Example:
// Let the user walk around on a grid and place out walls, as well as a starting square.
//  Pressing  enter  will  generate  a  path  between  the  starting  square  and  the  user's  current  location.
// All the keyboard commands are spoken at the start of the program.

const  int  maze_size=10;
// The maze is always a square, so setting this constant to 10 generates a 10 by 10 maze (100 squares).

//  Declare  the  two  dimensional  maze  array.
int[][]  maze;

//  Now  declare  the  variables  that  will  hold  the  user's  current  coordinates,  and  the  coordinates  of  the  starting  square.
int  x=0;
int  y=0;
int  start_x=-1;
int  start_y=-1;

//  Finally,  declare  the  pathfinder  and  tts_voice  objects.
pathfinder  holmes;
tts_voice  speech;

void  main()
{
show_game_window("Maze  Builder");

// Resize the grid to the proper dimentions and fill it with empty space.
maze.resize(maze_size);
for(int  i=0;i<maze_size;i++)
{
maze[i].resize(maze_size);
for(int  i2=0;i2<maze_size;i2++)
maze[i][i2]=0;
}

// Set up the pathfinder.
if(holmes.create_map(maze_size,  maze_size)==false)
{
speech.speak_wait("Could  not  set  up  the  map.");
exit();
}
holmes.set_callback_function(maze_callback);

speech.speak("Use the four arrow keys to navigate on the grid, and press escape to quit. Press space to place out a wall, and delete to remove one. Press s to set the starting square, and enter to ask the computer to walk from the starting square to your current location. Press the letter keys x and y to check your current coordinates. Press the d key to toggle whether diagonal paths are allowed or not.");

// Run the main game loop, which exits when the user presses escape.
while(key_down(KEY_ESCAPE)==false)
{

// Allow the user to move, but not outside the grid boundries.
if(key_pressed(KEY_LEFT))
{
x-=1;
if(x<0)
x+=1;
else
speak_location();
}
if(key_pressed(KEY_RIGHT))
{
x+=1;
if(x>=maze_size)
x-=1;
else
speak_location();
}
if(key_pressed(KEY_DOWN))
{
y-=1;
if(y<0)
y+=1;
else
speak_location();
}
if(key_pressed(KEY_UP))
{
y+=1;
if(y>=maze_size)
y-=1;
else
speak_location();
}

// Place out and remove walls.
if(key_pressed(KEY_SPACE))
{
if(maze[x][y]==1)
{
speech.speak_interrupt("Already  a  wall  there.");
}
else
{
maze[x][y]=1;
speech.speak_interrupt("Added  wall.");
}
}
if(key_pressed(KEY_DELETE))
{
if(maze[x][y]==0)
{
speech.speak_interrupt("No  wall  there.");
}
else
{
maze[x][y]=0;
speech.speak_interrupt("Removed  wall.");
}
}

// Find a path from the starting square to the user's current location.
if(key_pressed(KEY_RETURN))
{
if(start_x==-1  or  start_y==-1)
{
speech.speak_interrupt("Please  specify  a  starting  square  by  pressing  s  somewhere.");
continue;
}
vector[]  path=holmes.find(start_x,  start_y,  x,  y,  "");

if(path.length()==0)
{
speech.speak_interrupt("No  path  found.");
continue;
}

// We now have a path, so we assemble a string with the information that should be spoken.
string  output="";

// We now go through the path and convert the absolute coordinates to relative moves like west, north, etc.
int  last_x=start_x;
int  last_y=start_y;
for(int  i=0;i<path.length();i++)
{

//  Be  nice  to  speech  synthesizers.
if(output!="")
{
output+=",  ";
}

//  Check  for  the  8  possible  directions,  inserting  messages  in  the  string  accordingly.
// Note that only four of these will actually be present in the path unless diagonal paths are enabled.
if(path[i].x<last_x  and  path[i].y==last_y)
output+="west";
if(path[i].x>last_x  and  path[i].y==last_y)
output+="east";
if(path[i].y<last_y  and  path[i].x==last_x)
output+="south";
if(path[i].y<last_y  and  path[i].x<last_x)
output+="south  west";
if(path[i].y<last_y  and  path[i].x>last_x)
output+="south  east";
if(path[i].y>last_y  and  path[i].x==last_x)
output+="north";
if(path[i].y>last_y  and  path[i].x<last_x)
output+="north  west";
if(path[i].y>last_y  and  path[i].x>last_x)
output+="north  east";

last_x=path[i].x;
last_y=path[i].y;
}
speech.speak_interrupt(output);
}

//  Set  the  starting  location.
if(key_pressed(KEY_S))
{
start_x=x;
start_y=y;
speech.speak_interrupt("Starting square set to x " + x + ", y " + y + ".");
}

//  Should  we  allow  diagonal  paths?
if(key_pressed(KEY_D))
{
if(holmes.allow_diagonals==true)
{
holmes.allow_diagonals=false;
speech.speak_interrupt("Diagonal  paths  are  now  forbidden.");
}
else
{
holmes.allow_diagonals=true;
speech.speak_interrupt("Diagonal  paths  are  now  allowed.");
}
}

//  Check  our  coordinates.

if(key_pressed(KEY_X))
{
speech.speak_interrupt("X  "  +  x);
}
if(key_pressed(KEY_Y))
{
speech.speak_interrupt("Y  "  +  y);
}
wait(5);
}

speech.speak_interrupt_wait("Thanks  for,  playing?  I  guess?");
}

// This function tells the user whether the square is free or has a wall on it, and whether it is the starting square.
void  speak_location()
{
if(maze[x][y]==1)
{
if(x==start_x  and  y==start_y)
speech.speak_interrupt("Starting  square  and  wall.");
else
speech.speak_interrupt("Wall.");
return;
}
if(x==start_x  and  y==start_y)
speech.speak_interrupt("Starting  square  and  floor.");
else
speech.speak_interrupt("Floor.");
}

// This is the pathfinder callback. We simply check if there is a wall on the square and if there is, return 10. If not, return 0.
int  maze_callback(int  x,  int  y,  int  parent_x,  int  parent_y,  string  user_data)
{
if(maze[x][y]==1)
{
return  10;
}
return  0;
}

pathfinder object

This method cancels the current pathfinding operation.

bool cancel()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
If you cancel the current pathfinding operation, the find call will return an array with 0 elements and set the last error flag
accordingly.

This method may only be invoked from within the user callback.

Example:
See the main pathfinder chapter and the pathfinding tutorial.

pathfinder object

This method sets up the pathfinder for use with a map of a given size.

bool create_map(int x_size, int y_size)

Parameters:
x_size
The width of the map, in squares.
y_size
The height of the map, in squares.

Return value:
true on success, false on failure.

Remarks:
Before pathfinding can begin, the map needs to be set up. This method allocates resources for the entire map, which are used
internally while searching.

This method must not be invoked from within the user callback.

The maximum size of a map is 10000 by 10000 squares.

Example:
See the main pathfinder chapter and the pathfinding tutorial.

pathfinder object

This method destroys the current map so that no further search operations may be performed until a new one is set up.

bool destroy_map()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method frees the internal resources associated with the map. Note that this is done automatically when the pathfinder object
itself is destroyed.

This method must not be invoked from within the user callback.

Example:
See the main pathfinder chapter and the pathfinding tutorial.

pathfinder object

This method starts a new path search.

vector[] find(int source_x, int source_y, int destination_x, int destination_y, string user_data)

Parameters:
source_x
The x coordinate of the source square that the search starts from.
source_y
The y coordinate of the source square that the search starts from.
destination_x
The x coordinate of the destination square that must be reached.
destination_y
The y coordinate of the destination square that must be reached.
user_data
A string of arbitrary data that will be passed to the callback during the search.

Return value:
An array of vectors indicating the path on success, or an empty array on failure.

Remarks:
Before this method may be called, a map must have been set up and a user callback must have been specified. For more
information about the callback function such as its exact signature and its use, see the pathfinder_callback chapter.

The user data allows you to pass along special information to the callback that may be useful while searching. This string will
often contain a number which identifies the entity for whom the search is being performed, so that the callback may enforce
special rules for the type of creature in question etc.

This method must not be invoked from within the user callback of any pathfinder object.

Please note: You may neither call this method from the constructor of a class that is instanciated in global scope, nor from an
overloaded operator method.

Example:
See the main pathfinder chapter and the pathfinding tutorial.

pathfinder object

This method allows you to specify a function in your script that should be used as the pathfinder callback.

bool set_callback_function(pathfinder_callback@ callback)

Parameters:
callback
A handle to a function that conforms to the pathfinder_callback signature.

Return value:
true on success, false on failure.

Remarks:
For more information about the callback function such as its exact signature and its use, see the pathfinder_callback chapter.

This method must not be invoked from within the user callback.

Example:
See the main pathfinder chapter and the pathfinding tutorial.

pathfinder object

This user callback is invoked by the pathfinder for each square that it needs to gather information about during a search
operation.

int pathfinder_callback(int x, int y, int parent_x, int parent_y, string user_data)

Parameters:
x
The x coordinate of the square that the pathfinder is requesting information about.
y
The y coordinate of the square that the pathfinder is requesting information about.
parent_x
The x coordinate of the square from which the pathfinder went to reach the current square that it is now interested in.
parent_y
The y coordinate of the square from which the pathfinder went to reach the current square that it is now interested in.
user_data
The user data that was given in the call to the find method.

Return value:
The function must return a value between 0 and 10. Values between 0 and 9 are walkable squares, where higher numbers
indicate a greater risk associated with the given square. Returning 10 means that the square is unwalkable.

Remarks:
This function is invoked for each square that the pathfinder looks at when determining the best path to the destination. It is thus
very important that the function runs quickly, and that it does not wait for any events to occur or load resources from disk etc.

The user data allows you to pass along special information to the callback that may be useful while searching. This string will
often contain a number which identifies the entity for whom the search is being performed, so that the callback may enforce
special rules for the type of creature in question etc.

The parent coordinates are only useful in special circumstances. If, for instance, you are writing a two dimensional side scroller
and you want the entity who follows the path to be allowed to move downwards but not upwards, you can check if the x
coordinate is the same as the parent x coordinate and if the y coordinate is greater than the parent y coordinate. If this is the
case, you can return 10 to indicate that the square is unwalkable.

You may not call any methods of the pathfinder from within this callback function, except for the cancel method.

Example:
See the main pathfinder chapter and the pathfinding tutorial.

pathfinder object

int desperation_factor
This is a number between 0 and 9. It outweighs the cost to reach a particular square, so that the generated path may be more
costly (which is usually the same as more dangerous) but shorter if the desperation factor is high. With a low desperation factor,
the pathfinder will try to avoid squares that have a higher risk to them. For instance, if the desperation factor is 8 and the user
callback returns a risk of 9 for a particular square, the final cost for this square will be 1. Similarly if the desperation factor is 8
and the callback returns 5, the final cost is 0. The desperation factor is set to 0 by default.

bool allow_diagonals
This determines whether diagonal paths are allowed. The default is false.

int search_range
The search range is used to limit the area in which the pathfinder searches, which may be compared with a field of vision for the
entity following the path. If this is set to 0, the entire map will be searched. If the number is greater than 0, the pathfinder will only
find a path if the number of required steps is less than or equal to it. This is set to 0 by default.

bool active
Returns true while a map is set up (e.g. between calls to the create_map method and the destroy_map method), or false
otherwise. This property cannot be modified from the script.

settings object

The settings object is used to store data in the Windows registry.

settings()

Parameters:
None.

Remarks:
The settings object is useful for storing small amounts of data that can be retrieved when needed. Some common tasks include
storing and retrieving registration keys, game configurations such as music and sound volumes, and other game related data such
as current score, level, difficulty etc. However, the settings object should not be used to serialize game states. These should
always be stored in a file.

Please note: Caution should be exercised when accessing the registry. If done incorrectly, it can impair the performance of the
end user's system. Certain limitations apply when writing content to the registry, see the chapters for the individual functions for
more details.

Example:
// Store a value in the registry.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
success=game_data.write_number("music_volume",  -5.0);
if(!success)
{
alert("Error",  "Couldn't  write  music_volume.");
}
}

settings object

The settings object is used to store data in the Windows registry.

settings()

Parameters:
None.

Remarks:
The settings object is useful for storing small amounts of data that can be retrieved when needed. Some common tasks include
storing and retrieving registration keys, game configurations such as music and sound volumes, and other game related data such
as current score, level, difficulty etc. However, the settings object should not be used to serialize game states. These should
always be stored in a file.

Please note: Caution should be exercised when accessing the registry. If done incorrectly, it can impair the performance of the
end user's system. Certain limitations apply when writing content to the registry, see the chapters for the individual functions for
more details.

Example:
// Store a value in the registry.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
success=game_data.write_number("music_volume",  -5.0);
if(!success)
{
alert("Error",  "Couldn't  write  music_volume.");
}
}

settings object

This method will free all resources allocated for the settings object.

bool close()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
If the close method is not called explicitly, the object will automatically free itself when the object goes out of scope or when the
engine shuts down.

Example:
//  Store  a  value  in  the  registry  and  explicitly  close  afterwards.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
game_data.write_number("music_volume",  -5.0);
game_data.close();
}

settings object

This method will check to see if a particular value exists in your game's store in the Windows registry.

bool exists(string value_name)

Parameters:
value_name
The name of the value to look for.

Return value:
true if the value was found, false if it was not or if an error occured.

Remarks:
None.

Example:
//  Delete  our  music_volume  value  from  the  registry  if  it  exists.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
if(!game_data.exists("music_volume"))
{
alert("Error",  "Could  not  delete  music_volume.  The  specified  value  does  not  exist.");
exit();
}
success=game_data.remove_value("music_volume");
if(!success)
{
alert("Error",  "Could  not  delete  music_volume.");
exit();
}
}

settings object

This method will read the data contained in a numeric value.

double read_number(string value_name)

Parameters:
value_name
The name of the value to read.

Return value:
The value on success, or 0 on failure.

Remarks:
If an error occurs, the method will return 0 and the last_error flag will be set.

Example:
// Read the music volume from the registry.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
if(!game_data.exists("music_volume"))
{
alert("Error",  "Could  not  read  music_volume.  The  specified  value  does  not  exist.");
exit();
}
double  vol=game_data.read_number("music_volume");
if(last_error<0)
{
alert("Error",  "Could  not  access  music_volume.");
exit();
}
alert("Music  Volume",  vol);
}

settings object

This method will read the data contained in a string value.

string read_string(string value_name)

Parameters:
value_name
The name of the value to read.

Return value:
The value data on success, or a blank string on failure.

Remarks:
If an error occurs, the method will return a blank string and the last_error flag will be set.

Example:
// Read a user license from the registry.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
if(!game_data.exists("user_registration"))
{
alert("Error",  "Could  not  read  user_registration.  The  specified  value  does  not  exist.");
exit();
}
string  reg=game_data.read_string("user_registration");
if(last_error<0)
{
alert("Error",  "Could  not  access  user_registration.");
exit();
}
alert("User  Registration",  reg);
}

settings object

This method will delete a product's data store.

bool remove_product()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
If the product's store has any remaining values, these will also be removed.

If a company name is used and this is the last product within it, the company key is also removed.

If the games key contains no more companies or individual products, it too is removed.

Example:
// Delete our game from the registry.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
success=game_data.remove_product();
if(!success)
{
alert("Error",  "Could  not  remove  Bonebreaker's  data  store.");
exit();
}
}

settings object

This method will delete a value from your game's store in the Windows registry.

bool remove_value(string value_name)

Parameters:
value_name
The name of the value to delete.

Return value:
true on success, false on failure.

Remarks:
None.

Example:
//  Delete  our  music_volume  value  from  the  registry  if  it  exists.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
if(!game_data.exists("music_volume"))
{
alert("Error",  "Could  not  delete  music_volume.  The  specified  value  does  not  exist.");
exit();
}
success=game_data.remove_value("music_volume");
if(!success)
{
alert("Error",  "Could  not  delete  music_volume.");
exit();
}
}

settings object

This method will set the product and company if applicable to a game and allocate a store in the Windows registry for saving
data.

bool setup(string company, string product, bool all_users)

Parameters:
company
The name of the company if applicable.
product
The name of the product.
all_users
A boolean specifying whether the settings should be applicable for all user accounts on the system or the currently logged on user
only.

Return value:
true on success, false on failure.

Remarks:
All registry information created by BGT is stored in HKEY_LOCAL_MACHINE\Software\Games or
HKEY_CURRENT_USER\Software\Games.

The company and product parameters may be at most 128 bytes long, must contain at least 1 alphabetic character and no
unprintable characters.

If a company is supplied, the store will be located in the key company\product within the games key. If the company parameter
is empty, the product key will be directly in the games subkey.

If all_users is false, HKEY_CURRENT_USER is accessed as opposed to HKEY_LOCAL_MACHINE which is used if
all_users is true.

If an application is not running with administrative privileges and you have set up the settings to affect all users, this method will
return false, the last_error flag will be set and no reading, writing or deleting of values will be possible.

Example:
// Store a value in the registry.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
game_data.write_number("music_volume",  -5.0);
}

settings object

This method will write a numeric value to the registry.

bool write_number(string value_name, double content)

Parameters:
value_name
The name of the value to write.
content
The data to write to the specified value.

Return value:
True on success, false on failure.

Remarks:
Please note: You may not store more than 50 values in the registry for a given product.

Example:
// Write a user's score to the registry.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
success=game_data.write_number("userscore",  17408);
if(!success)
{
alert("Error",  "Could  not  write  userscore.");
exit();
}
}

settings object

This method will write a string value to the registry.

bool write_string(string value_name, string content)

Parameters:
value_name
The name of the value to write.
content
The data to write to the specified value.

Return value:
True on success, false on failure.

Remarks:
Please note: You may not store more than 50 values in the registry for a given product. The data length for each value may not
exceed 2048 bytes.

Example:
// Write a user's name to the registry.

void  main()
{
settings  game_data;
bool  success=game_data.setup("Testtime  Interactive",  "Bonebreaker",  false);
if(!success)
{
alert("Error",  "Could  not  access  the  registry.");
exit();
}
success=game_data.write_string("username",  "Percival  Bailey");
if(!success)
{
alert("Error",  "Could  not  write  username.");
exit();
}
}

settings object

bool active
This is true while a configuration session is in progress, false otherwise. Data can only be written while the session is active. This
property cannot be modified from the script.

string object

The string object is used to store text.

string()

Parameters:
None.

Remarks:
The string object contains overloaded operators which allow it to be assigned, compared and accessed like any other variable.
Additionally, it allows access to each individual character by way of indexes specified in square brackets, similar to arrays.

Strings are unique in that they can be declared and used as variables, or they can be passed back and forth as literal values.

Please see the String Variables chapter in the Language Tutorial for in-depth information regarding strings.

Example:
// Declare a string and display its length.

void  main()
{
string  name;
name="Damien";
alert("String  example",  "The  name  "+name+"  is  "+name.length()+"  characters  long.");
}

string object

The string object is used to store text.

string()

Parameters:
None.

Remarks:
The string object contains overloaded operators which allow it to be assigned, compared and accessed like any other variable.
Additionally, it allows access to each individual character by way of indexes specified in square brackets, similar to arrays.

Strings are unique in that they can be declared and used as variables, or they can be passed back and forth as literal values.

Please see the String Variables chapter in the Language Tutorial for in-depth information regarding strings.

Example:
// Declare a string and display its length.

void  main()
{
string  name;
name="Damien";
alert("String  example",  "The  name  "+name+"  is  "+name.length()+"  characters  long.");
}

string object

This method returns the length of the string.

uint get_size()

Parameters:
None.

Return value:
The length of the string, or 0 if the string is empty or if an error occurs.

Remarks:
Please note that the value returned is the length of the string rather than the index of the last character in the string. Therefore
when using the get_size method as a basis for a loop it is important that you check that any counting variable accessing individual
characters in the string stops when the counter reaches the length mark, not beyond it. This is because the index is 0 based. In
other words, with a string containing 3 entries you may access elements 0, 1 and 2.

Example:
// Declare a string and display its characters.

void  main()
{
string[]  hello;
hello="Hello";
alert("String",  "The  string  "+hello+"  contains  "  +  hello.length()  +  "  characters.");
for(int  counter=0;  counter<hello.get_size();  counter++)
{
alert("String",  "Character  "+counter+"  is  "+hello[counter]+".");
}
}

string object

This method is used to determine whether the given string is empty. That is to say, contains no characters.

bool is_empty()

Parameters:
None.

Return value:
True if the string is empty, false otherwise.

Remarks:
None.

Example:
// Declare a string and check if it is empty.

void  main()
{
string[]  example="Fantastic!";
if(example.is_empty())
{
alert("Oh dear!", "The string seems to be empty, despite attempting to assign a value to it.");
exit();
}
alert("Information",  "The  string  is  nicely  full  of  data.");
}

string object

This method returns the length of the string.

uint length()

Parameters:
None.

Return value:
The length of the string, or 0 if the string is empty or if an error occurs.

Remarks:
Please note that the value returned is the length of the string rather than the index of the last character in the string. Therefore
when using the length method as a basis for a loop it is important that you check that any counting variable accessing individual
characters in the string stops when the counter reaches the length mark, not beyond it. This is because the index is 0 based. In
other words, with a string containing 3 entries you may access elements 0, 1 and 2.

Example:
// Declare a string and display its characters.

void  main()
{
string[]  hello;
hello="Hello";
alert("String",  "The  string  "+hello+"  contains  "  +  hello.length()  +  "  characters.");
for(int  counter=0;  counter<hello.length();  counter++)
{
alert("String",  "Character  "+counter+"  is  "+hello[counter]+".");
}
}

string object

This method resizes a string.

void resize(uint new_length)

Parameters:
new_length
The new length of the string. A value of 0 means the string is reset completely.

Return value:
None.

Remarks:
If the new_length parameter is greater than the current length, the relevant number of null characters will be added to the string. If
the value is less, the string is truncated from right to left.

Example:
// Declare a string and display its values.

void  main()
{
string  hello;
hello="Hello";
hello.resize(hello.length()-1);
for(uint  counter=0;  counter<hello.length();  counter++)
{
alert("Hello",  "Hello  now  reads  "+hello);
}
}

sound object

The sound object is used to play audio stored either as Wav or Ogg, either in a folder or in a pack file, with or without
encryption.

sound()

Parameters:
None.

Remarks:
When a sound object is first created, it will not be active. To activate it you must call either the "load" or the "stream" method on
the object, specifying a file that should be associated with it. You may call these methods again at any time if you wish to
associate another sound with the object, at which point the old association (if any) will be cleared. This, in short, allows you to
reuse the same sound object for playing more than one sound.

Example:
// Load a sound into memory and play it.

void  main()
{
sound  test;
test.load("c:\\windows\\media\\ding.wav");
test.play_wait();
test.close();
}

sound object

The sound object is used to play audio stored either as Wav or Ogg, either in a folder or in a pack file, with or without
encryption.

sound()

Parameters:
None.

Remarks:
When a sound object is first created, it will not be active. To activate it you must call either the "load" or the "stream" method on
the object, specifying a file that should be associated with it. You may call these methods again at any time if you wish to
associate another sound with the object, at which point the old association (if any) will be cleared. This, in short, allows you to
reuse the same sound object for playing more than one sound.

Example:
// Load a sound into memory and play it.

void  main()
{
sound  test;
test.load("c:\\windows\\media\\ding.wav");
test.play_wait();
test.close();
}

sound object

This method will close any sound associated with the object.

bool close()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
Closing a sound is useful when you want to release the current sound memory that is associated with the object in order to save
resources, while still keeping the object alive for future use. This method is executed automatically when a sound object is
destroyed, so you need not worry about calling it to clean up after yourself at the end of your script for example. This method is
also executed if you call either the load or stream method on a sound object that already has a sound associated with it.

Example:
// Load a sound into memory, play it and close it.

void  main()
{
sound  test;
test.load("c:\\windows\\media\\ding.wav");
test.play_wait();
test.close();
}

sound object

This method will load a sound into memory for playback.

bool load(string filename)

Parameters:
filename
The file to open. If the sound is stored on disk this can be either an absolute path, a relative path or the file name on its own.
Absolute paths are not applicable however, if the sound is being loaded from a pack file (see remarks).

Return value:
true on success, false on failure.

Remarks:
The location in which the engine searches for the file specified in the filename parameter is determined by a call to the
"set_sound_storage" function. By default it will look on the users hard drive and in the directory where the program is stored,
unless an absolute path is specified such as "C:\\Windows\\media\\ding.wav". If the engine has been set up to look inside a pack
file then only a file name such as "creature.ogg" or a relative path such as "sounds\\monster.ogg" can be specified.

Note that both \ and / can be used to specify paths.

A loaded sound is completely put into memory when it is first created. This means that more of the system memory is taken up,
but this also lowers the CPU usage considerably as all the audio data is available at all times. Loaded sounds should be used for
most things that are meant to play quickly and/or often such as footsteps, gunshots, character noises and the like.

The engine uses a technique to try and ensure that the least possible amount of memory is used when the same sound is loaded
into more than one object, which is often the case with the previously mentioned sounds. For instance, if you have 50 enemies for
which you have two footsteps each, 100 footstep sounds will not be loaded into memory even though you get 100 completely
independent sound objects.

Example:
// Load a sound into memory and play it.

void  main()
{
sound  test;
test.load("c:\\windows\\media\\ding.wav");
test.play_wait();
test.close();
}

sound object

This method will set up the sound object for playback using data that is already stored in memory, as opposed to reading it from
a file on disk like the load method does.

bool load_from_memory(string data)

Parameters:
data
A string with the data that is to be loaded.

Return value:
true on success, false on failure.

Remarks:
The data must be a valid Ogg Vorbis or Wave stream, with or without encryption. If the data is encrypted, the engine will decrypt
it automatically using the global sound decryption key. The data is copied from your string into an internal buffer, so there is no
need to keep your string around after this function has returned.

A loaded sound is completely put into memory when it is first created. This means that more of the system memory is taken up,
but this also lowers the CPU usage considerably as all the audio data is available at all times. Loaded sounds should be used for
most things that are meant to play quickly and/or often such as footsteps, gunshots, character noises and the like.

Example:
// Store the contents of a sound file in a string, and then set up a sound object with this data.

void  main()
{
file  reader;
reader.open("c:\\windows\\media\\ding.wav",  "rb");
if(reader.active==false)
{
alert("Error",  "The  file  could  not  be  opened  for  reading.");
exit();
}
string  data=reader.read();
reader.close();
sound  test;
test.load_from_memory(data);
test.play_wait();
test.close();
}

sound object

This method will pause the sound playback.

bool pause()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method will stop the playback of the sound, without resetting its position to the beginning of the file. Thus, it will continue from the exact point at
which it was paused the next time the "play" method is called.

Example:
// Play a sound and let the user pause and unpause it with the space bar, stop it with enter and exit with escape.

void  main()
{
sound  ambience;
show_game_window("Sound  Test");
ambience.load("curry.wav");
if(ambience.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
ambience.play();
while(true)
{
if(key_pressed(KEY_SPACE))
{
if(ambience.playing==true)
{
ambience.pause();
}
else
{
ambience.play();
}
}
if(key_pressed(KEY_RETURN))
{
ambience.stop();
}
if(key_pressed(KEY_ESCAPE))
{
break;
}
wait(5);
}
}

sound object

This method will start the sound playback.

bool play()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method returns immediately after the sound has started playing, it does not wait for the playback to finish. In order to
determine when the sound has stopped, use the "playing" property.

Example:
// Play a sound, and then manually wait for it to stop before exiting.

void  main()
{
sound  ambience;
ambience.load("ding.ogg");
if(ambience.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
ambience.play();
while(ambience.playing==true)
{
wait(5);
}
}

sound object

This method will start the sound playback and set it to loop forever.

bool play_looped()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method will start the playback of the sound in a looped state, meaning that it keeps repeating until a method such as "pause"
or "stop" is called, or alternatively until the object is destroyed.

Example:
// Play a sound and loop it until the user presses space.

void  main()
{
sound  ambience;
ambience.load("ding.ogg");
show_game_window("Sound  Test");
if(ambience.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
ambience.play_looped();
while(key_pressed(KEY_SPACE)==false)
{
wait(5);
}
}

sound object

This method will start the sound playback and wait for it to finish before returning.

bool play_wait()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method will pause the entire script until the sound has finished playing, regardless of its length. If you wish to do other things
such as check for keystrokes while the sound is playing, call the "play" method instead and use the property called "playing" to
determine when the sound has stopped.

Example:
// Load a sound into memory, play it and close it.

void  main()
{
sound  test;
test.load("c:\\windows\\media\\ding.wav");
test.play_wait();
test.close();
}

sound object

This method will move the playing cursor in the sound to a new position.

bool seek(double new_position)

Parameters:
new_position
The position to seek to, specified in milliseconds. 0 indicates the beginning of the file.

Return value:
true on success, false on failure.

Remarks:
Seeking currently only works on static sounds, not streaming ones.

Example:
//Start playing a sound at the 1 second mark.

void  main()
{
sound  test;
test.load("curry.ogg");
test.seek(1000);
test.play_wait();
}

sound object

This method will stop the sound playback.

bool stop()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method will stop the playback of the sound, resetting its position to the beginning of the file. Thus, it will start from the very beginning the next time
the "play" method is called.

Example:
// Play a sound and let the user pause and unpause it with the space bar, stop it with enter and exit with escape.

void  main()
{
sound  ambience;
show_game_window("Sound  Test");
ambience.load("curry.wav");
if(ambience.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
ambience.play();
while(true)
{
if(key_pressed(KEY_SPACE))
{
if(ambience.playing==true)
{
ambience.pause();
}
else
{
ambience.play();
}
}
if(key_pressed(KEY_RETURN))
{
ambience.stop();
}
if(key_pressed(KEY_ESCAPE))
{
break;
}
wait(5);
}
}

sound object

This method will load a sound as a stream for playback.

bool stream(string filename)

Parameters:
filename
The file to open. If the sound is stored on disk this can be either an absolute path, a relative path or the file name on its own.
Absolute paths are not applicable however, if the sound is being loaded from a pack file (see remarks).

Return value:
true on success, false on failure.

Remarks:
The location in which the engine searches for the file specified in the filename parameter is determined by a call to the
"set_sound_storage" function. By default it will look on the users hard drive and in the directory where the program is stored,
unless an absolute path is specified such as "C:\\Windows\\media\\ding.wav". If the engine has been set up to look inside a pack
file then only a file name such as "creature.ogg" or a relative path such as "sounds\\monster.ogg" can be specified.

Note that both \ and / can be used to specify paths.

Be careful when creating streaming sounds. A streamed sound will be read from disk while it is playing which means that it does
not have to be loaded into memory in its entirety. This in turn means that the engine will constantly have to retrieve new data, so
playing many streams at once is not a good idea; especially not if they are in a compressed format which needs to be decoded on
the fly. Streames should generally be used for longer sounds such as background music, cut scenes or ambiences.

Example:
// Load a sound for streaming, play it and close it.

void  main()
{
sound  test;
test.stream("c:\\windows\\media\\ding.wav");
test.play_wait();
test.close();
}

sound object

double pan
This is the place in the stereo field in which the sound is positioned.
-100 is far left, 0 is center, and 100 is far right. This number may contain decimals, but will be rounded internally if needed and
so absolute values should not be relied upon. This is set to -1 on error.

double volume
This determines how loudly the sound will play. 0 is the original volume, and -100 is completely silent. This number may contain
decimals, but will be rounded internally if needed and so absolute values should not be relied upon. This is set to -1 on error.

double pitch
This is the pitch/speed at which the sound will play. 100 is the original, 50 is half speed, 1 is the lowest speed possible, 200 is
double speed, and 400 is quadruple speed. Note that different soundcards will impose different minimum's and maximum's for
this value so always check what the pitch actually turned out as after an assignment if you wish to be certain. This number may
contain decimals, but is very likely to be rounded internally which will result in an actual pitch that is very close to the intended
value but not exact. This is set to -1 on error.

double pitch_lower_limit
This specifies the lowest possible pitch value for the particular sound. The lower limit will differ depending on the sound's sample
rate, so always be sure to use this if you are trying to reach the minimum pitch. The pitch of the sound will never go below this
value. This is set to -1 on error. This property cannot be modified from the script.

bool active
This is set to true when a sound file is associated with the object, and to false otherwise. This property cannot be modified from
the script.

bool playing
This is set to true when the sound is playing, and to false when paused, stopped or on error. This property cannot be modified
from the script.

bool paused
This is set to true when the sound has been paused with an explicit call to the pause method, and to false otherwise and on error.
This property cannot be modified from the script.

double position
This is the current position in milliseconds of the playing cursor in the sound. This is set to -1 if the current position cannot be
retrieved. This property cannot be modified from the script.

double length
This is the total length of the sound in milliseconds. This is set to -1 if the length cannot be retrieved. This property cannot be
modified from the script.

double sample_rate
This is the sample rate of the current sound, in Hz. This is set to -1 on error. This property cannot be modified from the script.

double channels
This is the number of channels that the current sound contains. 1 is mono, 2 is stereo and -1 means an error has occurred. This
property cannot be modified from the script.

double bits
This is the bit depth of the current sound, usually 8 or 16, or -1 on error. This property cannot be modified from the script.

timer object

The timer object is used to measure elapsed time in milliseconds.

timer()

Parameters:
None.

Remarks:
Timer objects are mostly used inside loops in order to make something happen after a certain amount of time like a missile hitting its target, or continuously with a specified interval such as enemy
movement.

The idea is to set up a timer object and then check the value of the elapsed property over and over again until it has reached or exceeded the expected number of milliseconds. You never want to
check if the elapsed property matches the expected value exactly, as it is very likely that the timer goes past it by 1 or 2 milliseconds since your script usually needs to wait at least 1 millisecond in
every loop cycle as to avoid using 100 % of the processor's CPU. This does not apply to shorter loops that execute quickly, only to long-running ones such as the main game loop. In short, always
check whether the elapsed property is equal to or greater than your intended millisecond amount rather than exactly equal to it.

Please note that despite their high precision, timer objects do not slow down the script execution even when stored in large quantities. Thus, it is perfectly acceptable to use hundreds or even
thousands of timers at once without concern.

The timer will start running instantly when it is first created. If you have one or more timers as global variables, then this will happen while the script is still initializing. Thus, it might be a good idea to
call the restart method on these timers if you need to be sure exactly when the counting starts at a later point in your script.

Example:
// Play ding every 250MS using two buffers to avoid cutting the sound off as quickly, and let user press escape to quit and p to pause/unpause timer.

void  main()
{
sound  ding1;
sound  ding2;
show_game_window("Timer  Test");

ding1.load("C:\\Windows\\media\\ding.wav");
ding2.load("C:\\Windows\\media\\ding.wav");
if(ding1.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
if(ding2.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}

// We pan the two buffers a little bit so that they can be distinguished.
ding1.pan=-20;
ding2.pan=20;

int  current_buffer=1;
bool  paused=false;

ding1.play();

timer  counter;

while(true)
{
if(key_pressed(KEY_ESCAPE))
{
exit();

}
if(key_pressed(KEY_P))
{
if(paused==true)
{
paused=false;
counter.resume();
}
else
{
counter.pause();
paused=true;
}
}
if(counter.elapsed>=250)
{
counter.restart();
if(current_buffer==1)
{
current_buffer=2;
if(ding2.playing==true)
{
ding2.stop();
}
ding2.play();
}
else
{
current_buffer=1;
if(ding1.playing==true)
{
ding1.stop();
}
ding1.play();
}
}
wait(5);
}
}

timer object

The timer object is used to measure elapsed time in milliseconds.

timer()

Parameters:
None.

Remarks:
Timer objects are mostly used inside loops in order to make something happen after a certain amount of time like a missile hitting its target, or continuously with a specified interval such as enemy
movement.

The idea is to set up a timer object and then check the value of the elapsed property over and over again until it has reached or exceeded the expected number of milliseconds. You never want to
check if the elapsed property matches the expected value exactly, as it is very likely that the timer goes past it by 1 or 2 milliseconds since your script usually needs to wait at least 1 millisecond in
every loop cycle as to avoid using 100 % of the processor's CPU. This does not apply to shorter loops that execute quickly, only to long-running ones such as the main game loop. In short, always
check whether the elapsed property is equal to or greater than your intended millisecond amount rather than exactly equal to it.

Please note that despite their high precision, timer objects do not slow down the script execution even when stored in large quantities. Thus, it is perfectly acceptable to use hundreds or even
thousands of timers at once without concern.

The timer will start running instantly when it is first created. If you have one or more timers as global variables, then this will happen while the script is still initializing. Thus, it might be a good idea to
call the restart method on these timers if you need to be sure exactly when the counting starts at a later point in your script.

Example:
// Play ding every 250MS using two buffers to avoid cutting the sound off as quickly, and let user press escape to quit and p to pause/unpause timer.

void  main()
{
sound  ding1;
sound  ding2;
show_game_window("Timer  Test");

ding1.load("C:\\Windows\\media\\ding.wav");
ding2.load("C:\\Windows\\media\\ding.wav");
if(ding1.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
if(ding2.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}

// We pan the two buffers a little bit so that they can be distinguished.
ding1.pan=-20;
ding2.pan=20;

int  current_buffer=1;
bool  paused=false;

ding1.play();

timer  counter;

while(true)
{
if(key_pressed(KEY_ESCAPE))
{
exit();

}
if(key_pressed(KEY_P))
{
if(paused==true)
{
paused=false;
counter.resume();
}
else
{
counter.pause();
paused=true;
}
}
if(counter.elapsed>=250)
{
counter.restart();
if(current_buffer==1)
{
current_buffer=2;
if(ding2.playing==true)
{
ding2.stop();
}
ding2.play();
}
else
{
current_buffer=1;
if(ding1.playing==true)
{
ding1.stop();
}
ding1.play();
}
}
wait(5);
}
}

timer object

This method forces the timer to an absolute millisecond count.

bool force(double milliseconds)

Parameters:
milliseconds
The number of milliseconds to set the timer to.

Return value:
true on success, false on failure.

Remarks:
This method sets the elapsed time to an absolute value, in milliseconds. If the timer has previously been paused with a call to the
pause method, it will automatically be resumed.

Example:
// Create a timer and set it to make it seem as though 10 seconds have elapsed.

void  main()
{
timer  test;
test.force(10000);
alert("Result",  test.elapsed  +  "  milliseconds  have  now  elapsed.");
}

timer object

This method will pause the timer.

bool pause()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method makes the timer stop completely and remain at its current position until either the resume or the restart method is called on it. This is useful when you wish to create a pause feature in your
game for instance. Note that it will fail if the timer is already paused.

Example:
// Play ding every 250MS using two buffers to avoid cutting the sound off as quickly, and let user press escape to quit and p to pause/unpause timer.

void  main()
{
sound  ding1;
sound  ding2;
show_game_window("Timer  Test");

ding1.load("C:\\Windows\\media\\ding.wav");
ding2.load("C:\\Windows\\media\\ding.wav");
if(ding1.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
if(ding2.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}

// We pan the two buffers a little bit so that they can be distinguished.
ding1.pan=-20;
ding2.pan=20;

int  current_buffer=1;
bool  paused=false;

ding1.play();

timer  counter;

while(true)
{
if(key_pressed(KEY_ESCAPE))
{
exit();
}
if(key_pressed(KEY_P))
{
if(paused==true)
{
paused=false;
counter.resume();
}
else
{

counter.pause();
paused=true;
}
}
if(counter.elapsed>=250)
{
counter.restart();
if(current_buffer==1)
{
current_buffer=2;
if(ding2.playing==true)
{
ding2.stop();
}
ding2.play();
}
else
{
current_buffer=1;
if(ding1.playing==true)
{
ding1.stop();
}
ding1.play();
}
}
wait(5);
}
}

timer object

This method will restart the timer.

bool restart()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method resets the timer and makes it start again from the very beginning. If it has previously been paused with a call to the pause method, it will automatically be resumed.

Example:
// Play ding every 250MS using two buffers to avoid cutting the sound off as quickly, and let user press escape to quit and p to pause/unpause timer.

void  main()
{
sound  ding1;
sound  ding2;
show_game_window("Timer  Test");

ding1.load("C:\\Windows\\media\\ding.wav");
ding2.load("C:\\Windows\\media\\ding.wav");
if(ding1.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
if(ding2.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}

// We pan the two buffers a little bit so that they can be distinguished.
ding1.pan=-20;
ding2.pan=20;

int  current_buffer=1;
bool  paused=false;

ding1.play();

timer  counter;

while(true)
{
if(key_pressed(KEY_ESCAPE))
{
exit();
}
if(key_pressed(KEY_P))
{
if(paused==true)
{
paused=false;
counter.resume();
}
else
{
counter.pause();

paused=true;
}
}
if(counter.elapsed>=250)
{
counter.restart();
if(current_buffer==1)
{
current_buffer=2;
if(ding2.playing==true)
{
ding2.stop();
}
ding2.play();
}
else
{
current_buffer=1;
if(ding1.playing==true)
{
ding1.stop();
}
ding1.play();
}
}
wait(5);
}
}

timer object

This method will resume a timer that is paused.

bool resume()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method makes the timer continue from the exact point at which it was paused. Note that it will fail if the timer has not previously been paused with a call to the pause method.

Example:
// Play ding every 250MS using two buffers to avoid cutting the sound off as quickly, and let user press escape to quit and p to pause/unpause timer.

void  main()
{
sound  ding1;
sound  ding2;
show_game_window("Timer  Test");

ding1.load("C:\\Windows\\media\\ding.wav");
ding2.load("C:\\Windows\\media\\ding.wav");
if(ding1.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}
if(ding2.active==false)
{
alert("Error",  "The  sound  could  not  be  loaded.");
exit();
}

// We pan the two buffers a little bit so that they can be distinguished.
ding1.pan=-20;
ding2.pan=20;

int  current_buffer=1;
bool  paused=false;

ding1.play();

timer  counter;

while(true)
{
if(key_pressed(KEY_ESCAPE))
{
exit();
}
if(key_pressed(KEY_P))
{
if(paused==true)
{
paused=false;
counter.resume();
}
else
{
counter.pause();

paused=true;
}
}
if(counter.elapsed>=250)
{
counter.restart();
if(current_buffer==1)
{
current_buffer=2;
if(ding2.playing==true)
{
ding2.stop();
}
ding2.play();
}
else
{
current_buffer=1;
if(ding1.playing==true)
{
ding1.stop();
}
ding1.play();
}
}
wait(5);
}
}

timer object

double elapsed
This is the number of milliseconds that have elapsed since the timer object was first created, or since the last call to the restart
method. On error, it is set to -1. This property cannot be modified from the script.

bool running
This will be true when the timer is running, and false when it is paused or if an error occurs. This property cannot be modified
from the script.

tone_synth object

The tone_synth object is used to create tunes and store them as PCM wave sounds.

tone_synth()

Parameters:
None.

Remarks:
The idea is to create a tune, as simple or as complicated as you wish. The tune can have chords or single notes, with multiple or
single waveform types, and can be any length. When your tune is finished you can then create a wave file of your composition.

Please note that this is not an instrumental synth, but rather a waveform synth. This means that the sound output produced will be
similar to that of an older game console. The four basic waveform types supported are sine, square, sawtooth and triangle.

When creating a tune, lengths can either be specified in beats, or in milliseconds by using the ms methods. When specifying the
length in beats, the synth will automatically calculate the appropriate milliseconds to use based on the tempo property, which
specifies beats per minute.

To specify a length in beats, 1 specifies one beat, 0.5 is half a beat, 0.666 is a swing beat, etc.

Example:
// Make a simple tune and write it to a wave file.

tone_synth  synth;

void  main()
{
synth.tempo=120;
synth.waveform_type=2;
synth.note("C4",  4);
synth.note("E4",  4);
synth.note("G4",  4);
synth.waveform_type=3;
synth.note("C5",  0.666);
synth.rest(0.666);
synth.note("E5",  0.666);
synth.rest(0.666);
synth.note("G5",  0.666);
synth.rest(0.666);
synth.note("C6",  2);
synth.write_wave_file("synth.wav");
}

tone_synth object

The tone_synth object is used to create tunes and store them as PCM wave sounds.

tone_synth()

Parameters:
None.

Remarks:
The idea is to create a tune, as simple or as complicated as you wish. The tune can have chords or single notes, with multiple or
single waveform types, and can be any length. When your tune is finished you can then create a wave file of your composition.

Please note that this is not an instrumental synth, but rather a waveform synth. This means that the sound output produced will be
similar to that of an older game console. The four basic waveform types supported are sine, square, sawtooth and triangle.

When creating a tune, lengths can either be specified in beats, or in milliseconds by using the ms methods. When specifying the
length in beats, the synth will automatically calculate the appropriate milliseconds to use based on the tempo property, which
specifies beats per minute.

To specify a length in beats, 1 specifies one beat, 0.5 is half a beat, 0.666 is a swing beat, etc.

Example:
// Make a simple tune and write it to a wave file.

tone_synth  synth;

void  main()
{
synth.tempo=120;
synth.waveform_type=2;
synth.note("C4",  4);
synth.note("E4",  4);
synth.note("G4",  4);
synth.waveform_type=3;
synth.note("C5",  0.666);
synth.rest(0.666);
synth.note("E5",  0.666);
synth.rest(0.666);
synth.note("G5",  0.666);
synth.rest(0.666);
synth.note("C6",  2);
synth.write_wave_file("synth.wav");
}

tone_synth object

This method generates a given frequency or list of frequencies that will last for a specified number of beats.

bool freq(string frequencies, double length)

Parameters:
frequencies
A string specifying a frequency or list of frequencies to generate.
length
The length of the generated frequencies, in beats.

Return value:
true on success, false on failure.

Remarks:
Each frequency in the list is separated by a comma and space. Valid values are from 1 to 20000Hz. If the list does not meet the
criteria the function will fail.

The frequencies provided in the list will be generated together (as opposed to separately, one after the other), allowing for easier
and quicker creation of chords.

The length is specified in beats, based on the value of the tempo property.

Example:
// Write a C major chord using frequencies.

tone_synth  synth;

void  main()
{
synth.freq("262,  328,  394",  4);
synth.write_wave_file("wave.wav");
}

tone_synth object

This method generates a given frequency or list of frequencies that will last for a specified number of beats, with the additional
option of being able to bend.

bool freq_bend(string frequencies, double bend_amount, double length, double bend_start, double bend_length)

Parameters:
frequencies
A string specifying a frequency or list of frequencies to generate.
bend_amount
The amount in Hz to bend the pitch.
length
The length of the generated frequencies, in beats.
bend_start
The beat at which to start the bend.
bend_length
The length of the bend, in beats.

Return value:
true on success, false on failure.

Remarks:
Each frequency in the list is separated by a comma and space. Valid values are from 1 to 20000Hz. If the list does not meet the
criteria the function will fail.

The frequencies provided in the list will be generated together (as opposed to separately, one after the other), allowing for easier
and quicker creation of chords.

When specifying bend_amount, positive values bend upwards while negative values bend downwards.

It is important to note that all the frequencies in a list of multiple frequencies will be bending by the same amount and therefore the
end result will sound musically different to that of the start result. To ensure that the bend matches your intended chord, use the
note_bend method instead.

The length is specified in beats, based on the value of the tempo property.

Example:
// Bend a C major chord.

tone_synth  synth;

void  main()
{
synth.freq_bend("262,  328,  394",  60,  16,  5,  8);
synth.write_wave_file("wave.wav");
}

tone_synth object

This method generates a given frequency or list of frequencies that will last for a specified number of milliseconds, with the
additional option of being able to bend.

bool freq_bend_ms(string frequencies, double bend_amount, double length, double bend_start, double bend_length)

Parameters:
frequencies
A string specifying a frequency or list of frequencies to generate.
bend_amount
The amount in Hz to bend the pitch.
length
The length of the generated frequencies, in milliseconds.
bend_start
The millisecond at which to start the bend.
bend_length
The length of the bend, in milliseconds.

Return value:
true on success, false on failure.

Remarks:
Each frequency in the list is separated by a comma and space. Valid values are from 1 to 20000Hz. If the list does not meet the
criteria the function will fail.

The frequencies provided in the list will be generated together (as opposed to separately, one after the other), allowing for easier
and quicker creation of chords.

When specifying bend_amount, positive values bend upwards while negative values bend downwards.

It is important to note that all the frequencies in a list of multiple frequencies will be bending by the same amount and therefore the
end result will sound musically different to that of the start result. To ensure that the bend matches your intended chord, use the
note_bend_ms method instead.

Example:
// Bend a C major chord.

tone_synth  synth;

void  main()
{
synth.freq_bend_ms("262,  328,  394",  60,  8000,  3000,  4000);
synth.write_wave_file("wave.wav");
}

tone_synth object

This method generates a given frequency or list of frequencies that will last for a specified number of milliseconds.

bool freq_ms(string frequencies, double length)

Parameters:
frequencies
A string specifying a frequency or list of frequencies to generate.
length
The length of the generated frequencies, in milliseconds.

Return value:
true on success, false on failure.

Remarks:
Each frequency in the list is separated by a comma and space. Valid values are from 1 to 20000Hz. If the list does not meet the
criteria the function will fail.

The frequencies provided in the list will be generated together (as opposed to separately, one after the other), allowing for easier
and quicker creation of chords.

Example:
// Write a C major chord using frequencies.

tone_synth  synth;

void  main()
{
synth.freq_ms("262,  328,  394",  2000);
synth.write_wave_file("wave.wav");
}

tone_synth object

This method generates a given note or list of notes that will last for a specified number of beats.

bool note(string notes, double length)

Parameters:
notes
A string specifying a note or list of notes to generate.
length
The length of the notes, in beats.

Return value:
true on success, false on failure.

Remarks:
Notes are written inside a string and are separated by a comma and space. Values inside notes are case sensitive.

Notes are written as the note letter from A to G (notice that they are upper case), followed by the octave number. To sharpen a
note add a # after the note and before the number, and to flatten it use a b (lower case). Middle C is C4. Valid notes are from
C0 to D#8/Eb8.

The notes provided in the list will be generated together (as opposed to separately, one after the other), allowing for easier and
quicker creation of chords.

The length is specified in number of beats based on the beats per minute specified in the tempo property.

Example:
// Write a B major chord.

tone_synth  synth;

void  main()
{
synth.note("B3,  D#4,  Gb4",  4);
synth.write_wave_file("wave.wav");
}

tone_synth object

This method generates a given note or list of notes that will last for a specified number of beats, with the additional option of
being able to bend.

bool note_bend(string notes, double bend_amount, double length, double bend_start, double bend_length)

Parameters:
notes
A string specifying a note or list of notes to generate.
bend_amount
The amount in semitones to bend the pitch.
length
The length of the generated notes, in beats.
bend_start
The beat at which to start the bend.
bend_length
The length of the bend, in beats.

Return value:
true on success, false on failure.

Remarks:
Notes are written inside a string and are separated by a comma and space. Values inside notes are case sensitive.

Notes are written as the note letter from A to G (notice that they are upper case), followed by the octave number. To sharpen a
note add a # after the note and before the number, and to flatten it use a b (lower case). Middle C is C4. Valid notes are from
C0 to D#8/Eb8.

The notes provided in the list will be generated together (as opposed to separately, one after the other), allowing for easier and
quicker creation of chords.

When specifying bend_amount, positive values bend upwards while negative values bend downwards.

The length is specified in beats, based on the value of the tempo property.

Example:
// Write a B major chord and bend it up to an E major.

tone_synth  synth;

void  main()
{
synth.note_bend("B3,  D#4,  Gb4",  5,  4,  1,  1);
synth.write_wave_file("wave.wav");
}

tone_synth object

This method generates a given note or list of notes that will last for a specified number of milliseconds, with the additional option
of being able to bend.

bool note_bend_ms(string notes, double bend_amount, double length, double bend_start, double bend_length)

Parameters:
notes
A string specifying a note or list of notes to generate.
bend_amount
The amount in semitones to bend the pitch.
length
The length of the generated notes, in milliseconds.
bend_start
The millisecond at which to start the bend.
bend_length
The length of the bend, in milliseconds.

Return value:
true on success, false on failure.

Remarks:
Notes are written inside a string and are separated by a comma and space. Values inside notes are case sensitive.

Notes are written as the note letter from A to G (notice that they are upper case), followed by the octave number. To sharpen a
note add a # after the note and before the number, and to flatten it use a b (lower case). Middle C is C4. Valid notes are from
C0 to D#8/Eb8.

The notes provided in the list will be generated together (as opposed to separately, one after the other), allowing for easier and
quicker creation of chords.

When specifying bend_amount, positive values bend upwards while negative values bend downwards.

Example:
// Write a B major chord and bend it up to an E major.

tone_synth  synth;

void  main()
{
synth.note_bend("B3,  D#4,  Gb4",  5,  2000,  500,  500);
synth.write_wave_file("wave.wav");
}

tone_synth object

This method generates a given note or list of notes that will last for a specified number of milliseconds.

bool note_ms(string notes, double length)

Parameters:
notes
a string specifying a note or list of notes to generate.
length
The number of milliseconds the notes should last.

Return value:
true on success, false on failure.

Remarks:
Notes are written inside a string and are separated by a comma and space. Values inside notes are case sensitive.

Notes are written as the note letter from A to G (notice that they are upper case), followed by the octave number. To sharpen a
note add a # after the note and before the number, and to flatten it use a b (lower case). Middle C is C4. Valid notes are from
C0 to D#8/Eb8.

The notes provided in the list will be generated together (as opposed to separately, one after the other), allowing for easier and
quicker creation of chords.

Example:
// Write a B major chord.

tone_synth  synth;

void  main()
{
synth.note("B3,  D#4,  Gb4",  2000);
synth.write_wave_file("wave.wav");
}

tone_synth object

This method deletes and resets all data that is being held by the synth.

bool reset()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
When calling this methid, all notes are removed from the arrangement and all settings are reset to their default values.

Example:
// Make a C major chord in the bass and a spread out C major chord as the melody.

tone_synth  synth;

void  main()
{
synth.tempo=120;

//chord
synth.waveform_type=2;
synth.note("C4",  4);
synth.note("E4",  4);
synth.note("G4",  4);

//melody
synth.waveform_type=3;
synth.note("C5",  0.666);
synth.rest(0.666);
synth.note("E5",  0.666);
synth.rest(0.666);
synth.note("G5",  0.666);
synth.rest(0.666);
synth.note("C6",  2);

//write  wave  file  and  reset.
synth.write_wave_file("synth.wav");
synth.reset();
}

tone_synth object

This method inserts a rest or pause into the music for a specified number of beats.

bool rest(double beats)

Parameters:
beats
Number of beats to rest

Return value:
true on success, false on failure.

Remarks:
This method can also be used as a fast forward option.

As the tone synth supports chords, it is essential that you call this method before moving on to your next phrase, otherwise all notes will be mashed into one area.

Example:
// Make a C major chord in the bass and a spread out C major chord as the melody. This uses the rest method to spread the notes.

tone_synth  synth;

void  main()
{
synth.tempo=120;

//chord
synth.waveform_type=2;
synth.note("C4",  4);
synth.note("E4",  4);
synth.note("G4",  4);

//melody
synth.waveform_type=3;
synth.note("C5",  0.666);
synth.rest(0.666);
synth.note("E5",  0.666);
synth.rest(0.666);
synth.note("G5",  0.666);
synth.rest(0.666);
synth.note("C6",  2);
synth.write_wave_file("synth.wav");
}

tone_synth object

This method inserts a rest or pause into the music for a specified number of milliseconds.

bool rest_ms(double time)

Parameters:
time
Number of milliseconds to rest

Return value:
true on success, false on failure.

Remarks:
This method can also be used as a fast forward option.

As the tone synth supports chords, it is essential that you call this method or the rest method before moving on to your next phrase, otherwise all notes will be mashed into
one area.

Example:
// Make a C major chord in the bass and a spread out C major chord as the melody. This uses the rest method to spread the notes.

tone_synth  synth;

void  main()
{
synth.tempo=120;

//chord
synth.waveform_type=2;
synth.note_ms("C4",  2000);
synth.note_ms("E4",  2000);
synth.note_ms("G4",  2000);

//melody
synth.waveform_type=3;
synth.note_ms("C5",  333);
synth.rest_ms(333);
synth.note_ms("E5",  333);
synth.rest_ms(333);
synth.note_ms("G5",  333);
synth.rest_ms(333);
synth.note_ms("C6",  1000);
synth.write_wave_file("synth.wav");
}

tone_synth object

This method rewinds the cursor to a previous position.

bool rewind(double beats)

Parameters:
beats
Number of beats to rewind.

Return value:
true on success, false on failure.

Remarks:
Rewinding is useful if you want to go back to a certain position relative to where the cursor is at present.

Example:
// Write a chord, two melody notes, then rewind to add some more.

tone_synth  synth;

void  main()
{
synth.tempo=120;
synth.waveform_type=2;
synth.note("C4",  4);
synth.note("E4",  4);
synth.note("G4",  4);
synth.waveform_type=3;
synth.note("C5",  2);
synth.rest(2);
synth.note("C6",  2);
synth.rewind(1.333);
synth.note("E5",  0.666);
synth.rest(0.666);
synth.note("G5",  0.666);
synth.rest(0.666);
synth.write_wave_file("synth.wav");
}

tone_synth object

This method rewinds the cursor to a previous position in milliseconds.

bool rewind_ms(double time)

Parameters:
time
Number of milliseconds to rewind.

Return value:
true on success, false on failure.

Remarks:
Rewinding is useful if you want to go back to a certain position relative to where the cursor is at present.

Example:
// Write a chord, two melody notes, then rewind to add some more.

tone_synth  synth;

void  main()
{
synth.waveform_type=2;
synth.note_ms("C4",  2000);
synth.note_ms("E4",  2000);
synth.note_ms("G4",  2000);
synth.waveform_type=3;
synth.note_ms("C5",  1000);
synth.rest_ms(1000);
synth.note_ms("C6",  1000);
synth.rewind_ms(666);
synth.note_ms("E5",  333);
synth.rest_ms(333);
synth.note_ms("G5",  333);
synth.rest_ms(333);
synth.write_wave_file("synth.wav");
}

tone_synth object

This method moves the cursor to any position in beats.

bool seek(double new_position)

Parameters:
new_position
The position to seek to, in beats.

Return value:
true on success, false on failure.

Remarks:
Whereas the rest and rewind methods seek relative to your current position, this seeks to an actual position of the piece.
Therefore if you specify the position as 1.5, the cursor will move to position 1.5 as opposed to 1.5 beats previous or after the
current position.

Example:
// Write a chord, two melody notes, then rewind to add some more.

tone_synth  synth;

void  main()
{
synth.tempo=120;
synth.waveform_type=2;
synth.note("C4",  4);
synth.note("E4",  4);
synth.note("G4",  4);
synth.waveform_type=3;
synth.note("C5",  2);
synth.rest(2);
synth.note("C6",  2);
synth.seek(0.666);
synth.note("E5",  0.666);
synth.rest(0.666);
synth.note("G5",  0.666);
synth.rest(0.666);
synth.write_wave_file("synth.wav");
}

tone_synth object

This method moves the cursor to any position in milliseconds.

bool seek_ms(double new_position)

Parameters:
new_position
The position to seek to, in milliseconds.

Return value:
true on success, false on failure.

Remarks:
Whereas the rest and rewind methods seek relative to your current position, this seeks to an actual position of the piece.
Therefore if you specify the position as 1.5, the cursor will move to position 1.5 as opposed to 1.5 beats previous or after the
current position.

Example:
// Write a chord, two melody notes, then rewind to add some more.

tone_synth  synth;

void  main()
{
synth.waveform_type=2;
synth.note_ms("C4",  2000);
synth.note_ms("E4",  2000);
synth.note_ms("G4",  2000);
synth.waveform_type=3;
synth.note_ms("C5",  1000);
synth.rest_ms(1000);
synth.note_ms("C6",  1000);
synth.seek_ms(333);
synth.note_ms("E5",  333);
synth.rest_ms(333);
synth.note_ms("G5",  333);
synth.rest_ms(333);
synth.write_wave_file("synth.wav");
}

tone_synth object

This method defines the amount of fade time for subsequently generated notes.

bool set_edge_fades(double start_time, double end_time)

Parameters:
start_time
The length of time that the note will fade in.
end_time
The length of time that the note will fade out.

Return value:
true on success, false on failure.

Remarks:
There should always be at least a slight fade in all the tones to avoid unwanted clicking.

The fades can be useful for changing the characteristics of a sound. For example, changing the fade in changes the attack of the note, while
changing the fadeout alters the decay. A longer fade in will create a smooth note attack, while a short fade in creates a sharper sound. A
longer fadeout is useful for simulating a guitar or piano, while a shorter fadeout simulates an organ, etc.

If either the fade in or fade out parameters exceed the length of the note, the fades will automatically readjust themselves to fit the note's
length.

These parameters are set in milliseconds, no function to set them in beats is available at this time.

The default start time set by the engine is 8 milliseconds, and the default end time is 12 milliseconds.

Example:
// Make a C major chord in the bass and a spread out C major chord as the melody, simulating a toy piano.

tone_synth  synth;

void  main()
{
synth.tempo=120;
synth.set_edge_fades(1,  5000);  //the  length  will  automatically  adjust.

//chord
synth.waveform_type=2;
synth.note("C4",  4);
synth.note("E4",  4);
synth.note("G4",  4);

//melody
synth.waveform_type=3;
synth.note("C5",  4);
synth.rest(0.666);
synth.note("E5",  4);
synth.rest(0.666);
synth.note("G5",  4);
synth.rest(0.666);
synth.note("C6",  4);
synth.write_wave_file("synth.wav");
}

tone_synth object

This method writes the current data to a wave file.

bool write_wave_file(string filename)

Parameters:
filename
The filename to write to.

Return value:
true on success, false on failure.

Remarks:
The filename can either be an absolute or relative path. The file may take a while to be created, depending on the complexity of
the piece.

Example:
// Write a B major chord to a wave file.

tone_synth  synth;

void  main()
{
synth.note("B3,  D#4,  Gb4",  4);
synth.write_wave_file("wave.wav");
}

tone_synth object

This method writes the current data to a sound object in memory.

sound@ write_wave_sound()

Parameters:
None.

Return value:
A sound object with the data loaded into it on success, or an inactive sound object on failure.

Remarks:
This method writes the data directly to a sound object in memory, without using a file in any way. The sound object may take a
while to be created, depending on the complexity of the piece.

Example:
// Generate and play a B major chord.

tone_synth  synth;

void  main()
{
synth.note("B3,  D#4,  Gb4",  4);
sound@  output=synth.write_wave_sound();
if(output.active==false)
{
alert("Error",  "The  sound  object  could  not  be  created.\nReason:  "  +  get_last_error_text());
exit();
}
output.play_wait();
}

tone_synth object

double volume
This determines how loudly subsequently generated notes will be played. 0 is the original volume, and -100 is completely silent.
This number may contain decimals, but will be rounded internally if needed and so absolute values should not be relied upon.
This property can be modified from the script. This is set to -1 on error.

double tempo
This is used to set the speed of the song. Values are given in beats per minute. Default value is 120. This is set to -1 on error.

double pan
This controls the stereo panning (left to right) of subsequently generated notes. -100 is far left, 0 is the middle, and 100 is far
right. The default value is 0. This number may contain decimals, but will be rounded internally if needed and so absolute values
should not be relied upon. This property can be modified from the script. This is set to -1 on error.

double note_transpose
Allows subsequently generated notes to be transposed so that they play in a different key. Default value is 0, I.E. note C remains
C. This is set to -1 on error.

double freq_transpose
Allows frequencies to be transposed so that they play at different frequencies. Default value is 0, I.E. 262 remains 262. This is
set to -1 on error.

double waveform_type
Determines the type of tone that will be generated. 1=sawtooth, 2=square, 3=sine and 4=triangle. Default value is 1. This is set
to -1 on error.

double position
The position of the cursor in beats. This property cannot be directly modified from the script, however it can be modified by
using the seek, rest and rewind functions. This is set to -1 on error.

double position_ms
The position of the cursor in milliseconds. This property cannot be directly modified from the script, however it can be modified
by using the seek_ms, rest_ms and rewind_ms functions. This is set to -1 on error.

double length
The length of the current data in beats, or -1 on error. This property cannot be modified from the script.

double length_ms
The length of the current data in milliseconds, or -1 on error. This property cannot be modified from the script.

bool reverb
A boolean specifying whether reverb should be used. This is set to false by default. Please note that reverb is applied individually
to each tone that is generated, in order to allow for maximum flexibility. It may therefore take a long time to add reverb to a
complex piece. This processing time is mostly affected by the setting of the reverb_decay property (see below).

double reverb_room_size
A value between 0 and 100 specifying the auditory room size of the reverb which controls its depth. A room size of 80, for
instance, would simulate the reverb in a very large room, such as a hall. This is set to 50 by default.

double reverb_damping
A value between 0 and 100 that controls the damping of the reverb. The higher the value, the more muffled the reverb will
become. This is useful for producing effects comparable to that of a cave. This is set to 50 by default.

double reverb_dry_level
A value between 0 and 100 representing the volume of the original signal. This is set to 35 by default. 50 represents the original
volume.

double reverb_wet_level
A value between 0 and 100 representing the volume of the reverberated signal. This is set to 15 by default.

double reverb_decay
The room size determines how long it takes for the reverberation to finish naturally. However, the decay may be used to end the
reverberation prematurely and get a smooth fade. In short, it controls how much time the reverberation is actually allowed to
continue for after the original signal has ended. This is specified in milliseconds, and is set to 2500 by default.

tts_voice object

The tts_voice object is used to speak text using the Microsoft Speech API, more commonly known as Sapi.

tts_voice()

Parameters:
None.

Remarks:
This object will work with any Sapi 5 compatible voices you have installed on your system.

Example:
//Speak  some  text  using  Sapi.

tts_voice  speech;

void  main()
{
speech.speak_wait("I  am  Microsoft  Sapi.  How  do  you  do?");
}

tts_voice object

The tts_voice object is used to speak text using the Microsoft Speech API, more commonly known as Sapi.

tts_voice()

Parameters:
None.

Remarks:
This object will work with any Sapi 5 compatible voices you have installed on your system.

Example:
//Speak  some  text  using  Sapi.

tts_voice  speech;

void  main()
{
speech.speak_wait("I  am  Microsoft  Sapi.  How  do  you  do?");
}

tts_voice object

This method will list the names of all the available voices on the system.

string[] get_voice_names()

Parameters:
None.

Return value:
An array of names on success, or an empty array on failure.

Remarks:
The list of voices returned by this method is cached. The tts_voice object automatically retrieves a list of all the available voices
when it is first created, and this list is then kept internally for performance reasons. This means that if the user installs or uninstalls
voices while your game is running, this list will be out of date. You may refresh the internal cache manually by calling the
refresh_voice_list method.

Example:
// Show a series of message boxes with the names of the voices that are currently installed.

void  main()
{
tts_voice  speech;
string[]  list=speech.get_voice_names();
for(uint  counter=0;  counter<list.length();  counter++)
{
alert("Voice  "+counter,list[counter]);
}
}

tts_voice object

This method will refresh the list of available voices that is cached by this instance of the tts_voice object.

bool refresh_voice_list()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method is present for the sake of completeness and does not normally need to be called explicitly. It is useful if you want your game
to handle the rare case when users install or uninstall voices while the game is running and already speaking text using text to speech.

Each instance of the tts_voice object holds a cache of the voices that are available on the system, at the time when the instance was
created. This is done in order to keep the list intact so that the user does not unintentionally specify an index corresponding to a different
voice, in case the list of voices should happen to change during the lifetime of the object instance.

Example:
// Refresh the list of voices continuously, and print a message if the list changes.

tts_voice  speech;

timer  recheck;

void  main()
{
show_game_window("Voice  searcher");

string[]  voice_list=speech.get_voice_names();
recheck.restart();
while(key_pressed(KEY_ESCAPE)==false)
{
if(recheck.elapsed>=1000)
{
speech.refresh_voice_list();
recheck.restart();
string[]  new_list=speech.get_voice_names();
if(new_list.length()==voice_list.length())
continue;
if(new_list.length()<voice_list.length())
{
alert("Terror!",  "One  or  more  voices  have  disappeared  from  your  system!  Forever!  Honor  their  memory.");
}
else
{
alert("Joy!",  "One  or  more  new  voices  now  live  in  your  machine!  Welcome  them  warmly!");
}
voice_list=new_list;
}
wait(5);
}
}

tts_voice object

This method sets the voice that is to be used for subsequently spoken text.

bool set_current_voice(int voice)

Parameters:
voice
The ID of the voice that is to be used, as indicated by a call to the get_voice_names method.

Return value:
true on success, false on failure.

Remarks:
The voice ID's correspond to the index that each voice has in the array returned by the get_voice_names method. In short, if
Microsoft Sam is the first voice in the array returned by get_voice_names, then 0 would be specified as the argument to this
method in order to select Microsoft Sam.

The list of voices returned by the get_voice_names method is cached. The tts_voice object automatically retrieves a list of all the
available voices when it is first created, and this list is then kept internally for performance reasons. This means that if the user
installs or uninstalls voices while your game is running then the list, and as a result also the ID's that this method uses to refer to it,
will be out of date. You may refresh the internal cache manually by calling the refresh_voice_list method.

Example:
// Let all the voices present on the system, introduce themselves by name.

void  main()
{
tts_voice  speech;
string[]  list=speech.get_voice_names();
for(uint  counter=0;  counter<list.length();  counter++)
{
speech.set_current_voice(counter);
speech.speak_wait(list[counter]);
}
}

tts_voice object

This method will speak a given string.

bool speak(string text)

Parameters:
text
The text to speak.

Return value:
true on success, false on failure.

Remarks:
If Sapi is already speaking the text will be placed in the buffer queue and is the last string spoken, unless the buffer is cleared
beforehand with either the stop, speak_interrupt or speak_interrupt_wait methods.

Example:
// Add various text to a buffer.

tts_voice  speech;

void  main()
{
speech.speak("This  is  text1.");
speech.speak("This  is  text2.");
speech.speak("This  is  text3.");
while(speech.speaking)
{
wait(5);
}
}

tts_voice object

This method will speak a given string, interrupting all other queued strings.

bool speak_interrupt(string text)

Parameters:
text
The text to speak.

Return value:
true on success, false on failure.

Remarks:
All previous text queued by the speak or speak_interrupt method will be cleared, and the new text spoken straight away.

Example:
// Add various text to a buffer and interrupt it.

tts_voice  speech;

void  main()
{
speech.speak("This  is  text1.");
speech.speak("This  is  text2.");
speech.speak("This  is  text3.");
wait(300);
speech.speak_interrupt("This  is  text4  interrupting  all  previous  announcements.");
while(speech.speaking)
{
wait(5);
}
}

tts_voice object

This method will speak a given string, interrupting all other queued strings, and will pause execution until the string is spoken.

bool speak_interrupt_wait(string text)

Parameters:
text
The text to speak.

Return value:
true on success, false on failure.

Remarks:
All previous text queued by the speak or speak_interrupt method will be cleared, and the new text spoken straight away.

Because script execution is paused while the text inside the speak_interrupt_wait method is being spoken, it cannot be stopped
or interrupted.

Example:
// Add various text to a buffer and interrupt it.

tts_voice  speech;

void  main()
{
speech.speak("This  is  text1.");
speech.speak("This  is  text2.");
speech.speak("This  is  text3.");
wait(300);
speech.speak_interrupt_wait("This  is  text4  interrupting  all  previous  announcements.");
}

tts_voice object

This method will speak a given string into a Wave file, pausing the execution until the process is completed.

bool speak_to_file(string filename, string text)

Parameters:
filename
The name of the Wave file to be created.
text
The text to speak.

Return value:
true on success, false on failure.

Remarks:
None.

Example:
// Speak some text into a file.

tts_voice  speech;

void  main()
{
speech.speak_to_file("speech.wav",  "This  is  some  text,  which  will  be  spoken  into  a  file  called  speech.wav.");
}

tts_voice object

This method will speak a given string and will pause execution until the string is spoken.

bool speak_wait(string text)

Parameters:
text
The text to speak.

Return value:
true on success, false on failure.

Remarks:
Because script execution is paused while the text inside the speak_wait method is being spoken, it cannot be stopped or
interrupted.

Example:
//  Speak  some  text.

tts_voice  speech;

void  main()
{
speech.speak_wait("This  is  some  text.");
}

tts_voice object

This method will stop the voice from speaking and reset the text buffer.

bool stop()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This method is here for greater ease of use. The exact same task can be accomplished by passing an empty string to any of the
interrupt functions. This stops the speech and completely clears the text buffer, so any other text that is ready to be spoken is
deleted.

Example:
// Speak some text and stop it midway through.

tts_voice  speech;

void  main()
{
speech.speak("I  should  be  stopping  round  about  now.");
wait(500);
speech.stop();
}

tts_voice object

int volume
This determines how loudly the voice will speak. 0 is the original volume, and -100 is completely silent. This is set to -1 on error.
Please note that volume is not automatically set to 0 when it loads, but will instead reflect the volume that is set in the control
panel.

int rate
This is the speed at which the voice will speak. Values range from -10 (slowest) to 10 (fastest), where 0 is normal. This is set to
-1 on error. Please note that rate is not automatically set to 0 when it loads, but will instead reflect the speed that is set in the
control panel.

int pitch
This is the pitch at which the voice will speak. Values range from -10 (lowest) to 10 (highest, where 0 is normal. This is set to -1
on error. The pitch is always 0 initially. Please note that some text-to-speech engines do not support pitch, in which case this
property has no effect.

bool active
This is set to true if Sapi was initialised successfully, and to false otherwise. This property cannot be modified from the script.

bool speaking
This is set to true when Sapi is speaking, and to false when it is not, or if an error has occurred. This property cannot be modified
from the script.

vector object

The vector object is used to store a set of coordinates for a 2d or 3d environment.

vector(float x, float y, float z)

Parameters:
x
An optional float representing the x value of the vector.
y
A float representing the y value of the vector. This parameter must be filled in if an x value is specified.
z
An optional float representing the z value of the vector.

Remarks:
The parameters may simply be left out if desired. This will simply give values of 0 to all the coordinates.

This object contains the following overloaded operators:

== and != (compares two vectors)
=, +=, +, -=, - (uses arithmetic to add and subtract vectors to and from other vectors)
*=, /=, *, / (multiplies/divides a vector by a scale value, represented by a float)

Example:
// Create a vector object and display its values.

void  main()
{
vector  game_world(5,  2,  3);
alert("Information",  "Current  coordinates  are  "+game_world.x+",  "+game_world.y+",  "+game_world.z+".");
}

vector object

The vector object is used to store a set of coordinates for a 2d or 3d environment.

vector(float x, float y, float z)

Parameters:
x
An optional float representing the x value of the vector.
y
A float representing the y value of the vector. This parameter must be filled in if an x value is specified.
z
An optional float representing the z value of the vector.

Remarks:
The parameters may simply be left out if desired. This will simply give values of 0 to all the coordinates.

This object contains the following overloaded operators:

== and != (compares two vectors)
=, +=, +, -=, - (uses arithmetic to add and subtract vectors to and from other vectors)
*=, /=, *, / (multiplies/divides a vector by a scale value, represented by a float)

Example:
// Create a vector object and display its values.

void  main()
{
vector  game_world(5,  2,  3);
alert("Information",  "Current  coordinates  are  "+game_world.x+",  "+game_world.y+",  "+game_world.z+".");
}

vector object

This method returns the length of the vector.

float length()

Parameters:
None.

Return value:
The total length of the vector.

Remarks:
None.

Example:
// Create a vector and display its length.

void  main()
{
vector  test(5,2,3);
alert("Information",  "The  vector's  length  is  "+test.length()+".");
}

vector object

float x
The current value of x on the vector.

float y
The current value of y on the vector.

float z
The current value of z on the vector.

Helper Layer

The helper layer consists of functions and objects that are included as part of the BGT distribution, but that are stored as
separate scripts that should be included in a main script. These are high level commonly used game tasks based around the
foundation layer and their use is entirely optional. As their name suggests, they are there to help the task of coding your game
become less painful and taxing.

Examples of helper objects include dynamic_menu, for creating a game menu, and number_speaker, a class that will concatenate
number files in whatever way necessary in order to speak numbers as smoothly and naturally as possible. Helper functions
include functions to simplify the positioning of sound sources in an environment, etc.

Helper Layer

The helper layer consists of functions and objects that are included as part of the BGT distribution, but that are stored as
separate scripts that should be included in a main script. These are high level commonly used game tasks based around the
foundation layer and their use is entirely optional. As their name suggests, they are there to help the task of coding your game
become less painful and taxing.

Examples of helper objects include dynamic_menu, for creating a game menu, and number_speaker, a class that will concatenate
number files in whatever way necessary in order to speak numbers as smoothly and naturally as possible. Helper functions
include functions to simplify the positioning of sound sources in an environment, etc.

position_sound_1d

This function is contained within sound_positioning.bgt in the BGT include directory, and is used to position an object based on the player's position in a
sidescrolling environment.

void position_sound_1d(sound@ handle, int listener_x, int source_x, float pan_step, float volume_step)

Parameters:
handle
the handle of a sound to position.
listener_x
The position of the player on the game board.
source_x
The position of the object on the game board.
pan_step
The value that specifies how much the sound should pan when the object and/or listener position changes.
volume_step
The value that specifies how much the sound's volume will change when the object and/or listener position changes.

Return value:
None.

Remarks:
This function will automatically adjust the volume and pan of a sound by comparing the values of listener_x and source_x and calculating the final change using
pan_step and volume_step, simulating a person moving towards or away from the source of a sound, usually a physical object, such as a monster or river.

It is important to remember that the listener value simulates the position of a person, and the source simulates the position of the object that is making the sound. If
these values are the same, the sound will be central and at full volume. If the listener position is lower than that of the object, the sound will be panned to the right at
an appropriate volume. If the listener value is higher than that of the object, the listener has passed the object and the sound will therefore come from the left, etc.

Example:
//Put a machine on square 20, allowing the player to use the arrow keys to walk left and right, hearing the effect.

#include  "sound_positioning.bgt"

int  p_position=0,  m_position=20,  f_position=40;  //declare  and  assign  player,  machine  and  finishing  positions  respectively.
sound  machine;  //sound  of  the  machine.

void  main()
{
machine.load("sounds/machine.wav");
if(machine.active==false)
{
alert("Error",  "sounds/machine.wav  could  not  be  loaded.");
}
show_game_window("Sound  position  test");
position_sound_1d(machine,  p_position,  m_position,  1,  1);
machine.play_looped();
while(true)
{
if(key_pressed(KEY_LEFT))
{
if(p_position  >  0)
{
p_position--;
position_sound_1d(machine,  p_position,  m_position,  1,  1);
}
}
if(key_pressed(KEY_RIGHT))
{

if(p_position  <  f_position)
{
p_position++;
position_sound_1d(machine,  p_position,  m_position,  1,  1);
}
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
wait(5);
}
}

position_sound_2d

This function is contained within sound_positioning.bgt in the BGT include directory, and is used to position an object based on
the player's position in an x/y-based grid.

void position_sound_2d(sound@ handle, int listener_x, int listener_y, int source_x, int source_y, float pan_step, float
volume_step, float behind_pitch_decrease)

Parameters:
handle
the handle of a sound to position.
listener_x
The X position of the player on the game board.
listener_y
The Y position of the player on the game board.
source_x
The X position of the object on the game board.
source_y
The Y position of the object on the game board.
pan_step
The value that specifies how much the sound should pan when the object and/or listener position changes.
volume_step
The value that specifies how much the sound's volume will change when the object and/or listener position changes.
behind_pitch_decrease
The amount to decrease the pitch by that will simulate the object sound behind the listener, relative to the current pitch.

Return value:
None.

Remarks:
This function will simulate a listener moving towards or away from an object in an X/Y-based environment. This means that the
user may walk forwards, backwards, left or right away from the sound. To simulate the object's position behind the listener, the
pitch can be changed to notify the player.

It is important to remember that the listener values simulates the position of a person, and the source simulates the position of the
object that is making the sound. If the X and Y values are the same, the sound will be central and at full volume. If the X values
change, the pan and volume values will change. If the Y value is changed, the volume will change.

Example:
//Put the machine on X 20, Y 5, allowing the user to use arrow keys to hear the effect.

#include  "sound_positioning.bgt"

int  p_position_x=0,  p_position_y=0;  //declare  and  assign  the  player's  positions
int  m_position_x=20,  m_position_y=5;  //declare  and  assign  the  machine's  position
int  f_position_x=40,  f_position_y=10;  //declare  and  assign  the  boundaries  of  the  grid
sound  machine;  //sound  of  the  machine

void  main()
{
machine.load("sounds/machine.wav");
if(machine.active==false)
{
alert("Error",  "sounds/machine.wav  could  not  be  loaded.");
}
show_game_window("Sound  position  test");
position_sound_2d(machine,  p_position_x,  p_position_y,  m_position_x,  m_position_y,  1,  1,  10);
machine.play_looped();

while(true)
{
if(key_pressed(KEY_LEFT))
{
if(p_position_x  >  0)
{
p_position_x--;
position_sound_2d(machine,  p_position_x,  p_position_y,  m_position_x,  m_position_y,  1,  1,  10);
}
}
if(key_pressed(KEY_RIGHT))
{
if(p_position_x  <  f_position_x)
{
p_position_x++;
position_sound_2d(machine,  p_position_x,  p_position_y,  m_position_x,  m_position_y,  1,  1,  10);
}
}
if(key_pressed(KEY_UP))
{
if(p_position_y  <  f_position_y)
{
p_position_y++;
position_sound_2d(machine,  p_position_x,  p_position_y,  m_position_x,  m_position_y,  1,  1,  10);
}
}
if(key_pressed(KEY_DOWN))
{
if(p_position_y  >  0)
{
p_position_y--;
position_sound_2d(machine,  p_position_x,  p_position_y,  m_position_x,  m_position_y,  1,  1,  10);
}
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
wait(5);
}
}

position_sound_custom_1d

This function is contained within sound_positioning.bgt in the BGT include directory, and is used to position an object based on the player's position in a
sidescrolling environment, allowing the default pan and volume values to be changed.

void position_sound_custom_1d(sound@ handle, int listener_x, int source_x, float pan_step, float volume_step, float start_pan, float start_volume)

Parameters:
handle
the handle of a sound to position.
listener_x
The position of the player on the game board.
source_x
The position of the object on the game board.
pan_step
The value that specifies how much the sound should pan when the object and/or listener position changes.
volume_step
The value that specifies how much the sound's volume will change when the object and/or listener position changes.
start_pan
The starting pan of the sound.
start_volume
The starting volume of the sound.

Return value:
None.

Remarks:
This function will automatically adjust the volume and pan of a sound by comparing the values of listener_x and source_x and calculating the final change using
pan_step and volume_step, simulating a person moving towards or away from the source of a sound, usually a physical object, such as a monster or river.

It is important to remember that the listener value simulates the position of a person, and the source simulates the position of the object that is making the sound. If
these values are the same, the sound will be central and at full volume. If the listener position is lower than that of the object, the sound will be panned to the right at
an appropriate volume. If the listener value is higher than that of the object, the listener has passed the object and the sound will therefore come from the left, etc.

Example:
//Put a machine on square 20, allowing the player to use the arrow keys to walk left and right, hearing the effect.

#include  "sound_positioning.bgt"

int  p_position=0,  m_position=20,  f_position=40;  //declare  and  assign  player,  machine  and  finishing  positions  respectively.
sound  machine;  //sound  of  the  machine.

void  main()
{
machine.load("sounds/machine.wav");
if(machine.active==false)
{
alert("Error",  "sounds/machine.wav  could  not  be  loaded.");
}
show_game_window("Sound  position  test");
position_sound_custom_1d(machine,  p_position,  m_position,  1,  1,  0,  -1);
machine.play_looped();
while(true)
{
if(key_pressed(KEY_LEFT))
{
if(p_position  >  0)
{
p_position--;

position_sound_custom_1d(machine,  p_position,  m_position,  1,  1,  0,  -1);
}
}
if(key_pressed(KEY_RIGHT))
{
if(p_position  <  f_position)
{
p_position++;
position_sound_custom_1d(machine,  p_position,  m_position,  1,  1,  0,  -1);
}
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
wait(5);
}
}

position_sound_custom_2d

This function is contained within sound_positioning.bgt in the BGT include directory, and is used to position an object based on the player's position
in an x/y-based grid, allowing the default pan, volume and pitch to be changed.

void position_sound_custom_2d(sound@ handle, int listener_x, int listener_y, int source_x, int source_y, float pan_step, float volume_step, float
behind_pitch_decrease, float start_pan, float start_volume, float start_pitch)

Parameters:
handle
the handle of a sound to position.
listener_x
The X position of the player on the game board.
listener_y
The Y position of the player on the game board.
source_x
The X position of the object on the game board.
source_y
The Y position of the object on the game board.
pan_step
The value that specifies how much the sound should pan when the object and/or listener position changes.
volume_step
The value that specifies how much the sound's volume will change when the object and/or listener position changes.
behind_pitch_decrease
The amount to decrease the pitch by that will simulate the object sound behind the listener, relative to the current pitch.
start_pan
The starting pan of the sound.
start_volume
The starting volume of the sound.
start_pitch
The starting pitch of the sound.

Return value:
None.

Remarks:
This function will simulate a listener moving towards or away from an object in an X/Y-based environment. This means that the user may walk
forwards, backwards, left or right away from the sound. To simulate the object's position behind the listener, the pitch can be changed to notify the
player.

It is important to remember that the listener values simulate the position of a person, and the source simulates the position of the object that is
making the sound. If the X and Y values are the same, the sound will be central and at full volume. If the X values change, the pan and volume
values will change. If the Y value is changed, the volume will change.

Example:
//Put the machine on X 20, Y 5, allowing the user to use arrow keys to hear the effect.

#include  "sound_positioning.bgt"

int  p_position_x=0,  p_position_y=0;  //declare  and  assign  the  player's  positions
int  m_position_x=20,  m_position_y=5;  //declare  and  assign  the  machine's  position
int  f_position_x=40,  f_position_y=10;  //declare  and  assign  the  boundaries  of  the  grid
sound  machine;  //sound  of  the  machine

void  main()
{
machine.load("sounds/machine.wav");

if(machine.active==false)
{
alert("Error",  "sounds/machine.wav  could  not  be  loaded.");
}
show_game_window("Sound  position  test");
position_sound_custom_2d(machine,  p_position_x,  p_position_y,  m_position_x,  m_position_y,  1,  1,  10,  0,  -1,  90);
machine.play_looped();
while(true)
{
if(key_pressed(KEY_LEFT))
{
if(p_position_x  >  0)
{
p_position_x--;
position_sound_custom_2d(machine,  p_position_x,  p_position_y,  m_position_x,  m_position_y,  1,  1,  10,  0,  -1,  90);
}
}
if(key_pressed(KEY_RIGHT))
{
if(p_position_x  <  f_position_x)
{
p_position_x++;
position_sound_custom_2d(machine,  p_position_x,  p_position_y,  m_position_x,  m_position_y,  1,  1,  10,  0,  -1,  90);
}
}
if(key_pressed(KEY_UP))
{
if(p_position_y  <  f_position_y)
{
p_position_y++;
position_sound_custom_2d(machine,  p_position_x,  p_position_y,  m_position_x,  m_position_y,  1,  1,  10,  0,  -1,  90);
}
}
if(key_pressed(KEY_DOWN))
{
if(p_position_y  >  0)
{
p_position_y--;
position_sound_custom_2d(machine,  p_position_x,  p_position_y,  m_position_x,  m_position_y,  1,  1,  10,  0,  -1,  90);
}
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
wait(5);
}
}

convert_to_currency

This function is contained within finance.bgt in the BGT include directory, and is used to convert a number into a valid currency
formatted string.

string convert_to_currency(int number, string currency_symbol)

Parameters:
number
The number that will be converted.
currency_symbol
Your local currency symbol, for example £, $, ¥, € etc.

Return value:
A string containing the converted currency.

Remarks:
The number parameter should contain the value in the smallest possible units. For example, if you are working in British Pounds
then you will be providing the value in pence, in dollars the value will be in cents, etc. The function will then convert it into pounds
and pence, dollars and cents, etc.

This function only works with positive numbers. Any value below 0 will return an empty string.

Example:
//Display  a  number  as  a  currency.

#include  "finance.bgt"

void  main()
{
alert("convert_to_currency  test",  convert_to_currency(150,  "£"));  //output  should  be  £1.50.
}

audio_form object

The audio_form object is stored in form.bgt in the BGT include directory, and is used to create a form-based interface that users
can interact with.

audio_form()

Parameters:
None.

Remarks:
The audio_form object is audio-based, using Microsoft SAPI 5.1 or a screen reader as its output, and relies on keyboard input.
It will simulate the functionality of a graphical interface, but there are no controls on the screen, and they can not be used with the
mouse.

An audio_form object can hold up to 50 controls.

Example:
// Make a simple form.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Cancel");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

The audio_form object is stored in form.bgt in the BGT include directory, and is used to create a form-based interface that users
can interact with.

audio_form()

Parameters:
None.

Remarks:
The audio_form object is audio-based, using Microsoft SAPI 5.1 or a screen reader as its output, and relies on keyboard input.
It will simulate the functionality of a graphical interface, but there are no controls on the screen, and they can not be used with the
mouse.

An audio_form object can hold up to 50 controls.

Example:
// Make a simple form.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Cancel");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

The following is a list of the supported control types for the audio_form object, as returned by the get_control_type method.

ct_button (0)
ct_input (1)
ct_checkbox (2)
ct_progress (3)
ct_status_bar (4)
ct_list (5)

audio_form object

Due to the complexity of the audio_form class, it is necessary for it to have its own error codes, which are independent to BGT's
errors. These can be obtained with the form's get_last_error method.
Below is a list of the constants and what they mean.

form_error_none (0) - Success.
form_error_invalid_index (1) - The specified control index is invalid.
form_error_invalid_control (2) - The specified control does not support the attempted operation. This could happen, for
example, if you attempted to get the progress percentage from a checkbox.
form_error_invalid_value (3) - A specified value is invalid. This occurs if a progress percentage is out of range.
form_error_invalid_operation (4) - The requested operation is invalid. This occurs if an operation has been carried out
once and cannot be done again at the present time.
form_error_no_window (5) - No window is present (the form is not active).
form_error_window_full (6) - No more controls can be added to the window, the limit has been reached.
form_error_text_too_long (7) - The text of an input box is longer than the maximum_length parameter.
form_error_list_empty (8) - A listbox is empty. This usually happens if you attempt to delete or clear an empty listbox.
form_error_list_full (9) - The listbox is full, the limit has been reached.
form_error_invalid_list_index (10) - The specified list index is invalid.
form_error_control_invisible (11) - The specified control is invisible. This happens if you try to set focus to an invisible
control.
form_error_no_controls_visible (12) - All controls are invisible, therefore none can be focused.

audio_form object

The following is a list of constants which are used to control the keyboard echo settings for input boxes.

textflag_none (0)
textflag_characters (1)
textflag_words (2)
textflag_characters_words (3)

audio_form object

These constants are used in conjunction with the edit_text method.

edit_mode_replace (0) - The remainder of the text will be deleted and replaced with the substitute in the new_text
parameter.
edit_mode_trim_to_length (1) - If the position of the edit and the length of the new text means the old text may become
longer, the remainder of the replacement is truncated to fit the current length of the original.
edit_mode_append_to_end (2) - All the replacement text is added in the specified position of the original regardless of
length, unless the replacement means the new text may go over the maximum length of a control.

audio_form object

This method will activate the speech timer of a progress bar.

bool activate_progress_timer(int control_id)

Parameters:
control_id
The control ID of the progress bar you wish to activate.

Return value:
true on success, false otherwise.

Remarks:
This function must be called if you intend the speak_interval parameter to be effective. When you are ready to start the operation
that requires the use of the progress bar, the timer must be activated and the progress set whenever the need arises.

Example:
// Make a simple form with a few buttons and a dummy progress bar.

#include  "form.bgt"

audio_form  form;
int  dummy_counter;

void  main()
{
dummy_counter=0;
form.create_window("Example  Form",  true);
int  dummy=form.create_progress_bar("&Copying",  5,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
form.activate_progress_timer(dummy);
while(true)
{
form.monitor();
wait(5);
dummy_counter++;
if(dummy_counter>=250)
{
if(form.get_progress(dummy)>=100)
{
form.pause_progress_timer(dummy);
}
else
{
form.set_progress(dummy,  form.get_progress(dummy)+1);
}
dummy_counter=0;
}
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will add an item to a listbox.

bool add_list_item(int control_id, string option, int position=-1, bool selected=false)

Parameters:
control_id
The control ID of the listbox.
option
The text of the item that is to be added.
position
An optional parameter specifying the 0-based position of the item. A value of -1 will append the item to the end of the list.
selected
An optional parameter indicating whether the item should be selected as part of a multiselect listbox.

Return value:
true on success, false on failure.

Remarks:
If a new item is to take the place of an old item, the old item, along with the rest of the list, will be moved to make room for the
new item. No existing items will be overwritten under any circumstances. If the maximum limit of a list has been reached, an error
will be flagged and the method will return false.

On a listbox that does not have multiselection capabilities, the selected flag is ignored.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}

}

audio_form object

This method will add or append text to an input box or status bar without interfering with any text that is already present.

bool add_text(int control_id, string text, int position=-1)

Parameters:
control_id
The control ID of the input box.
text
The text that is to be added.
position
An optional parameter specifying the 0-based position where the new text should be added. A value of -1 will append the text to
the end of the original.

Return value:
true on success, false on failure.

Remarks:
None.

Example:
// Make a simple form with a few buttons and an input box.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  input=form.create_input_box("test",  "Let's  add!",  "",  0,  true,  false);
int  ok=form.create_button("&Add");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
form.add_text(input,  "  text",  9);
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will clear a listbox, deleting all items.

bool clear_list(int control_id)

Parameters:
control_id
The control ID of the listbox.

Return value:
true on success, false on failure.

Remarks:
None.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
form.clear_list(list);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will create a push button and return a control ID.

int create_button(string caption, bool primary=false, bool cancel=false, bool overwrite=true)

Parameters:
caption
The text for the push button.
primary
An optional parameter which specifies whether this button should be the default on the form.
cancel
An optional parameter which specifies whether this button is the cancel button on the form.
overwrite
An optional parameter indicating whether certain button attributes should be overwritten (see remarks).

Return value:
A control ID that can be used to check the status of the control, or -1 on error.

Remarks:
Prepending any letter in the caption string with an ampersand (&) sign will cause a shortcut to be created for the button. For
example, Eξt will cause alt+x to be the shortcut for the exit button.

The primary and cancel parameters specify how the button should be handled in certain situations. A primary or default button
specifies that this should be activated if the Return key is pressed on any control other than a button or an editable multiline input
box, while a cancel button is activated when the Escape key is pressed. Thus, each of these attributes can only apply to any one
button at a time. If you choose to turn one or both of these attributes on, and any other button has these attributes set, you can
choose whether they are overwritten or kept in tact. If you decide not to overwrite attributes and it turns out another button does
indeed have these attributes set, the new button will still be created, but the attributes that are already set elsewhere will not be
applied. Please note that it is possible to have one button with both attributes set.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK",  true,  false);
int  cancel=form.create_button("Eξt",  false,  true);
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}

}

audio_form object

This method will create a checkbox and return a control ID.

int create_checkbox(string caption, bool initial_value=false, bool read_only=false)

Parameters:
caption
The text for the checkbox.
initial_value
An optional parameter specifying the initial value for the checkbox. true is checked, false is unchecked.
read_only
An optional parameter indicating whether the checkbox should be read only. true means the checkbox will be read only, false
means it can be modified.

Return value:
A control ID that can be used to check the status of the control, or -1 on error.

Remarks:
Prepending any letter in the caption string with an ampersand (&) sign will cause a shortcut to be created for the checkbox. For
example, &Subscribe will cause alt+s to be the shortcut for the Subscribe checkbox.

Example:
// Make a simple form with a few buttons and a checkbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  subscribe=form.create_checkbox("&Subscribe",  false,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will create a text field and return a control ID.

int create_input_box(string caption, string default_text="", string password_mask="", int max_length=0, bool read_only=false,
bool multiline=false)

Parameters:
caption
The title for the text field.
default_text
An optional parameter specifying the default text that is to be shown initially.
password_mask
An optional parameter specifying the character that is to be used to hide the characters of the text (if any). Passing an empty
string will cause all characters to be shown.
maximum_length
An optional parameter specifying the maximum length of the text field. A value of 0 means there is no set limit to the number of
characters that can be entered.
read_only
An optional parameter indicating whether the text field is to be read only. true means the control is read only, false means the
control can be modified.
multiline
An optional parameter indicating whether the control is to be multiline. true means the control is multiline, false means that it can
only accept one line.

Return value:
A control ID that can be used to check the status of the control, or -1 on error.

Remarks:
Prepending any letter in the caption string with an ampersand (&) sign will cause a shortcut to be created for the text field. For
example, &Name will cause alt+n to be the shortcut for the Name field.

If a password_mask parameter contains more than one character, the form will use the first character only.

Example:
// Make a simple form with a few buttons and a text field.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  name=form.create_input_box("&Name");
int  password=form.create_input_box("&Password",  "",  "*");
int  message=form.create_input_box("&Message",  "This  is  a  read  only  input  box.",  "",  0,  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{

exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will create a listbox and return a control ID.

int create_list(string caption, int maximum_items=0, bool multiselect=false)

Parameters:
caption
The title for the listbox.
maximum_items
An optional parameter specifying how many items can be in the listbox. A value of 0 means there is no limit to the number of
items that can be shown.
multiselect
An optional parameter indicating whether the listbox should have the capability to be able to manage multiple selections.

Return value:
A control ID that can be used to check the status of the control, or -1 on error.

Remarks:
Prepending any letter in the caption string with an ampersand (&) sign will cause a shortcut to be created for the listbox. For
example, &People will cause alt+p to be the shortcut for the list of people.

If the listbox is a multiselect-enabled listbox, each item in the list can be manipulated like a checkbox with the space bar, and
Control+A can be used to select the entire list.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will create a progress bar and return a control ID.

int create_progress_bar(string caption, int speak_interval=5, bool speak_global=true)

Parameters:
caption
The text for the progress bar.
speak_interval
An optional parameter specifying the interval at which the progress bar should notify the user, in seconds. A value of 0 means the
notification is not timed and the user will only be notified when the progress changes.
speak_global
An optional parameter indicating whether the progress is to be spoken globally. true means the progress will be announced
globally, false means the announcements will only be made if the progress bar is in focus.

Return value:
A control ID that can be used to check the status of the control, or -1 on error.

Remarks:
Prepending any letter in the caption string with an ampersand (&) sign will cause a shortcut to be created for the progress bar.
For example, &Copying will cause alt+c to be the shortcut for the progress bar.

Please note that only one progress bar can be set as global at a time. This is to avoid confusion with multiple progress bars
announcing different values. If the global flag is set and a progress bar is already determined to be announcing globally, the flag
will be set to false.

Example:
// Make a simple form with a few buttons and a dummy progress bar.

#include  "form.bgt"

audio_form  form;
int  dummy_counter;

void  main()
{
dummy_counter=0;
form.create_window("Example  Form",  true);
int  dummy=form.create_progress_bar("&Copying",  0,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
dummy_counter++;
if(dummy_counter>=250)
{
form.set_progress(dummy,  form.get_progress(dummy)+1);
dummy_counter=0;
}
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))

{
exit();
}
}
}

audio_form object

This method will create a status bar and return a control ID.

int create_status_bar(string caption, string text)

Parameters:
caption
The title of the status bar.
text
The text of the status bar.

Return value:
A control ID that can be used to check the status of the control, or -1 on error.

Remarks:
Prepending any letter in the caption string with an ampersand (&) sign will cause a shortcut to be created for the status bar. For
example, &Operation will cause alt+o to be the shortcut for the status bar.

Example:
// Make a simple form with a few buttons and a status bar.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  status=form.create_status_bar("&Operation",  "This  program  is  working  properly.");
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will create a window and activate the form.

void create_window(string window_title, bool change_screen_title=true, bool say_dialog=true)

Parameters:
window_title
The title of the window.
change_screen_title
An optional parameter determining whether the title should be changed on screen. If it is, the onscreen window will be created/
changed to that name.
say_dialog
An optional parameter determining whether the speech source should say the word "dialog" after the window title.

Return value:
none.

Remarks:
As all operations are performed instantaneously, the window must be created first before any controls can be created or used.

An audio_form object can hold up to 50 controls.

Example:
// Make a simple form.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form");
int  ok=form.create_button("OK",  true,  false);
int  cancel=form.create_button("Cancel",  false,  true);
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will delete a specified control.

bool delete_control(int control_id)

Parameters:
control_id
The ID of the control you wish to delete.

Return value:
true on success, false otherwise.

Remarks:
none.

Example:
// Make a simple form with a few buttons. The delete button will delete the OK button.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  delete=form.create_button("&Delete  Control");
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if(form.is_pressed(delete))
{
form.delete_control(ok);
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will delete an existing item from a listbox.

bool delete_list_item(int control_id, int list_index, bool reset_cursor=true, bool speak_highlighted_item=true)

Parameters:
control_id
The control ID of the listbox.
list_index
The list index of the item to delete.
reset_cursor
An optional value specifying whether the cursor should be reset so that no item is highlighted. If set to false, the cursor will move
to the previous item.
speak_highlighted_item
An optional parameter specifying whether speech should notify about the deleted item and announce the next highlighted item. If
set to false, speech will remain silent.

Return value:
true on success, false on failure.

Remarks:
None.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
form.delete_list_item(list,  2);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will delete all the selected items from a listbox.

bool delete_list_selections(int control_id, bool reset_cursor=true, bool speak_deletion_status=true)

Parameters:
control_id
The control ID of the listbox.
reset_cursor
An optional value specifying whether the cursor should be reset so that it is not highlighting an item. If set to false, the cursor will
either stay put, or move to the next available item.
speak_deletion_status
An optional parameter specifying whether speech should notify about the deleted items and announce the next highlighted item. If
set to false, speech will remain silent.

Return value:
true on success, false on failure.

Remarks:
If more than one item is selected in the listbox, the speech will only state the number of items deleted, as opposed to the text of
the item that is announced if only one item is deleted.

If this method is called on a listbox that is not set to accept multiple selections, the currently highlighted item will be deleted only.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0,  true);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
int  delete=form.create_button("Delete");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(delete))
{
form.delete_list_highlights(list);
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will edit an item in a listbox.

bool edit_list_item(int control_id, string new_option, int position)

Parameters:
control_id
The control ID of the listbox.
new_option
The new text for the item.
position
The position of the item to edit.

Return value:
true on success, false on failure.

Remarks:
None.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
form.edit_list_item(list,  "Damien",  0);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will edit text in an input box or status bar.

bool edit_text(int control_id, string new_text, int position, int edit_mode)

Parameters:
control_id
The control ID of the listbox.
new_text
The new text for the edited section.
position
The zero-based starting position for the edit.
edit_mode
One of the three available edit modes (see the text edit mode constants for further information).

Return value:
true on success, false on failure.

Remarks:
None.

Example:
// Make a simple form with an input box that edits when the change button is pressed.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("test",true);
int  text=form.create_input_box("Type  here",  "This  is  a  default  text",  "",  0,  true,  false);
int  btn1=form.create_button("&Change",true,false);
int  btn2=form.create_button("Eξt",false,true);
while(true)
{
form.monitor();
if(form.is_pressed(btn1))
{
form.edit_text(text,  "document",  18,  edit_mode_append_to_end);
}
if(form.is_pressed(btn2))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
wait(5);
}
}

audio_form object

This method will focus a specified control.

bool focus(int control_id)

Parameters:
control_id
The ID of the control you wish to focus.

Return value:
true on success, false otherwise.

Remarks:
None.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
form.focus(ok);
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the ID of the control that contains the cancel button.

int get_cancel_button()

Parameters:
None.

Return value:
The ID of the control on success, or -1 on failure or if no control has the cancel button.

Remarks:
For a more detailed explanation on how the primary and cancel attributes work, please see the topic covering the create_button method.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  set_device=form.create_button("Set  &Device");
int  reset=form.create_button("Reset  &Password");
int  website=form.create_button("Open  &Website");
int  ok=form.create_button("OK",  true,  false);
int  cancel=form.create_button("Eξt",  false,  true);
int  assist_me=form.create_button("&Help");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  cancel  button  is  "+form.get_cancel_button());
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if(form.is_pressed(assist_me))
{
alert("Help",  "This  program  does  pretty  much  nothing.  It  is  supposed  to  be  an  example  from  the  BGT  manual.");
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the caption of a given control.

string get_caption(int control_id)

Parameters:
control_id
The ID of the control to retrieve the caption from.

Return value:
The caption text on success, or an empty string on failure.

Remarks:
The caption text that is returned does not contain the & shortcut symbol. To retrieve the character which contains the shortcut,
use the get_shortcut method.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  caption  of  the  listbox  is  "+form.get_caption(list));
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return a textual representation of the attributes for a given control.

string get_control_attributes(int control_id)

Parameters:
control_id
The control ID to retrieve the attributes from.

Return value:
The control attributes on success, or an empty string on failure.

Remarks:
The text that this method returns is the text that the form uses to announce the control types to the end user.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  control  type  of  OK  is  "+form.get_control_attributes(ok));
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the number of controls currently on the window.

int get_control_count()

Parameters:
None.

Return value:
The control count on success, or 0 on failure.

Remarks:
None.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "There  are  currently  "+form.get_control_count()+"  controls  on  the  window.");
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the type of a given control (See control type constants).

int get_control_type(int control_id)

Parameters:
control_id
The control ID to retrieve the type from.

Return value:
The control type on success, or -1 on failure.

Remarks:
This function is useful for checking if you can perform a specific operation on the control.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  control  type  of  OK  is  "+form.get_control_type(ok));
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the ID of the control that is currently in focus.

int get_current_focus()

Parameters:
None.

Return value:
The ID of the focused control on success, or -1 on failure or if no control is in focus.

Remarks:
None.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  currently  focused  control  is  "+form.get_current_focus());
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the ID of the control that contains the primary/default button.

int get_default_button()

Parameters:
None.

Return value:
The ID of the control on success, or -1 on failure or if no control has the primary/default button.

Remarks:
For a more detailed explanation on how the primary and cancel attributes work, please see the topic covering the create_button method.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  set_device=form.create_button("Set  &Device");
int  reset=form.create_button("Reset  &Password");
int  website=form.create_button("Open  &Website");
int  ok=form.create_button("OK",  true,  false);
int  cancel=form.create_button("Eξt",  false,  true);
int  assist_me=form.create_button("&Help");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  default  button  is  "+form.get_default_button());
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if(form.is_pressed(assist_me))
{
alert("Help",  "This  program  does  pretty  much  nothing.  It  is  supposed  to  be  an  example  from  the  BGT  manual.");
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the error code for the last operation.

int get_last_error()

Parameters:
None.

Return value:
The Error code of the previous operation.

Remarks:
Every time a method is called, the error is reset. Therefore you must call the get_last_error method directly after the call to the
method you are checking.

Example:
// Make a simple form with a few buttons and cause a deliberate error.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
form.get_text(ok);
alert("Information",  "The  error  code  is  "+form.get_last_error());
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the number of items in a listbox.

int get_list_count(int control_id)

Parameters:
control_id
The control ID of the listbox.

Return value:
The number of items on success, -1 on failure.

Remarks:
This method gives the number of items on a listbox, rather than the index of the final item. Therefore if you are looping through
the list, you want to make sure that your counter is less than this value.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  listbox  has  "+form.get_list_count(list)+"  items.");
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the text of a specified list index.

string get_list_item(int control_id, int list_index)

Parameters:
control_id
The control ID of the listbox.
list_index
The index of the item to retrieve the text for.

Return value:
The text of the item on success, or an empty string on failure.

Remarks:
This method is useful for grabbing the text of the current cursor position (see example).

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  name  selected  is  "+form.get_list_item(list,  form.get_list_position(list)));
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the position of the cursor in a listbox.

int get_list_position(int control_id)

Parameters:
control_id
The control ID of the listbox.

Return value:
The cursor position on success, -1 on failure or if the cursor is not focused on a value.

Remarks:
This is useful for acting based on the item that is currently highlighted on the listbox in question.

Please note that the position does not need to be manually set, since the audio_form class changes the position automatically
when different items are selected.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  listbox  cursor  is  at  position  "+form.get_list_position(list));
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return a list of the selected item's positions in a listbox.

int[] get_list_selections(int control_id)

Parameters:
control_id
The control ID of the listbox.

Return value:
An array containing a list of selected items on success, or an empty array on failure.

Remarks:
This is useful for looping through all the selected items and acting based on the results.

If no items are selected, the following rules apply:

1.
2.

If the cursor is highlighting an item, the method will return the position of the cursor, as the only element in the array.
If the cursor is not highlighting an item, an empty array will be returned.

If you need to check for the success or failure of this method, it is necessary to check the error flag by using the get_last_error
method, since the returned array can be empty in various situations, even when the method has successfully done its work.

Please note that selections are automatically set by the audio_form and do not need to be manually set.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
string  text="The  current  highlights  are:  ";
int[]  selections;
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0,  true);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
selections=form.get_list_selections(list);
for(int  counter=0;  counter<selections.length();  counter++)
{
text+=selections[counter];
if(counter==(selections.length()-1))
{
text+=".";
}
else

{
text+=",  ";
}
}
alert("Information",  text);
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the value of a progress bar.

int get_progress(int control_id)

Parameters:
control_id
The control ID to retrieve the progress from.

Return value:
The progress value on success, or -1 on failure.

Remarks:
This method will only function on progress bars.

Example:
// Make a simple form with a few buttons and a dummy progress bar.

#include  "form.bgt"

audio_form  form;
int  dummy_counter;

void  main()
{
dummy_counter=0;
form.create_window("Example  Form",  true);
int  dummy=form.create_progress_bar("&Copying",  0,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
dummy_counter++;
if(dummy_counter>=250)
{
form.set_progress(dummy,  form.get_progress(dummy)+1);
dummy_counter=0;
}
if(form.is_pressed(ok))
{
alert("Information",  "The  current  value  of  the  progress  bar  is  "+form.get_progress(dummy));
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will return the text of a given control.

string get_text(int control_id)

Parameters:
control_id
The control ID to retrieve the text from.

Return value:
The text on success, or an empty string on failure.

Remarks:
The text that is retrieved is any additional text of the control, for example that found in text fields or status bars. To retrieve the
control's title, use the get_caption method.

Example:
// Make a simple form with a few buttons and a text field.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  test=form.create_input_box("Test",  "diddy  daddy  dumbo",  "",  0,  false,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  text  entered  in  the  text  field  is  "+form.get_text(test));
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method returns a value indicating whether the specified control is checked.

bool is_checked(int control_id)

Parameters:
control_id
The ID of the control you wish to check.

Return value:
true if the control is checked, false if the control is unchecked or if an error occurred.

Remarks:
This method will only function on checkboxes.

Example:
// Make a simple form with a few buttons and a checkbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  check=form.create_checkbox("Check  or  uncheck  me  as  you  wish",  false,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
if(form.is_checked(check))
{
alert("Information",  "The  checkbox  is  checked.");
}
else
{
alert("Information",  "The  checkbox  is  unchecked.");
}
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method returns a value indicating whether the specified control is enabled.

bool is_enabled(int control_id)

Parameters:
control_id
The ID of the control you wish to check.

Return value:
true if the control is enabled, false if the control is disabled or if an error occurred.

Remarks:
The enabled/disabled flag refers to whether the control may be manipulated in any way. If set to disabled, the audio feedback
will announce this and the control is unusable. If you wish the control to be unreachable, use the visibility status instead.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
if(form.is_enabled(ok))
{
alert("Information",  "The  OK  button  is  enabled.");
}
else
{
alert("Information",  "The  OK  button  is  disabled.");
}
exit();
}
if(form.is_pressed(cancel))
{
if(form.is_enabled(cancel))
{
alert("Information",  "The  Cancel  button  is  enabled.");
}
else
{
alert("Information",  "The  Cancel  button  is  disabled.");
}
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method returns a value indicating whether the specified control is a multiline text field.

bool is_multiline(int control_id)

Parameters:
control_id
The ID of the control you wish to check.

Return value:
true if the control is a multiline text field, false if the control is not a multiline text field or if an error occurred.

Remarks:
This method will only function on text fields.

Example:
// Make a simple form with a few buttons and a text field.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  test=form.create_input_box("Test",  "",  "",  0,  false,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
if(form.is_multiline(test))
{
alert("Information",  "The  text  field  is  multiline.");
}
else
{
alert("Information",  "The  text  field  is  not  multiline.");
}
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method returns a value indicating whether the specified control has been pressed.

bool is_pressed(int control_id)

Parameters:
control_id
The ID of the control you wish to check.

Return value:
true if the control has been pressed, false if the control has not been pressed or if an error occurred.

Remarks:
This method will only function on push buttons.

Once this method returns true, the button's push status is automatically reset so that the button may be reused.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  OK  button  has  been  pressed.");
exit();
}
if(form.is_pressed(cancel))
{
alert("Information",  "The  Cancel  button  has  been  pressed.");
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method returns a value indicating whether the specified control is read only.

bool is_read_only(int control_id)

Parameters:
control_id
The ID of the control you wish to check.

Return value:
true if the control is read only, false if the control can be modified, or if an error occurred.

Remarks:
This method will only function on text fields and checkboxes.

Example:
// Make a simple form with a few buttons and a text field.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  test=form.create_input_box("Test",  "",  "",  0,  false,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
if(form.is_read_only(test))
{
alert("Information",  "The  text  field  is  read  only.");
}
else
{
alert("Information",  "The  text  field  can  be  modified.");
}
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method returns a value indicating whether the specified control is visible.

bool is_visible(int control_id)

Parameters:
control_id
The ID of the control you wish to check.

Return value:
true if the control is visible, false if the control is invisible or if an error occurred.

Remarks:
In the case of the audio form, visibility simply refers to whether the control is accessible whilst tabbing through the dialog. If it is
set to invisible, the form's cursor will skip the control, but the control may still be used and/or modified by the script.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
if(form.is_visible(ok))
{
alert("Information",  "The  OK  button  is  visible.");
}
else
{
alert("Information",  "The  OK  button  is  invisible.");
}
exit();
}
if(form.is_pressed(cancel))
{
if(form.is_visible(cancel))
{
alert("Information",  "The  Cancel  button  is  visible.");
}
else
{
alert("Information",  "The  Cancel  button  is  invisible.");
}
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will monitor the form for events.

void monitor()

Parameters:
None.

Return value:
None.

Remarks:
This method must be called at regular intervals, as it performs various checks, mainly keyboard activity, to make the form and the
focused control work as expected.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will pause an active speech timer of a progress bar.

bool pause_progress_timer(int control_id)

Parameters:
control_id
The control ID of the progress bar you wish to pause.

Return value:
true on success, false otherwise.

Remarks:
This function must be called once you have finished using the progress bar.

Example:
// Make a simple form with a few buttons and a dummy progress bar.

#include  "form.bgt"

audio_form  form;
int  dummy_counter;

void  main()
{
dummy_counter=0;
form.create_window("Example  Form",  true);
int  dummy=form.create_progress_bar("&Copying",  5,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
form.activate_progress_timer(dummy);
while(true)
{
form.monitor();
wait(5);
dummy_counter++;
if(dummy_counter>=250)
{
if(form.get_progress(dummy)>=100)
{
form.pause_progress_timer(dummy);
}
else
{
form.set_progress(dummy,  form.get_progress(dummy)+1);
}
dummy_counter=0;
}
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will reset the form and cause it to become inactive.

void reset()

Parameters:
None.

Return value:
none.

Remarks:
None.

Example:
// Make a simple form.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Cancel");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
form.reset();
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will set the primary and cancel attributes on a push button.

bool set_button_attributes(int control_id, bool primary, bool cancel, bool overwrite=true)

Parameters:
control_id
The ID of the control you wish to change.
primary
A boolean value indicating whether the button should be the primary or default.
cancel
A boolean value indicating whether the button is the cancel button.

Return value:
true on success, false on failure.

Remarks:
For a more detailed explanation on how the primary and cancel attributes work, please see the topic covering the create_button
method.

Please note that the method will only return false if the method is unable to function due to an error. If the method at least
attempted to change the attributes, it will still return true. To check whether the attributes were changed, use the
get_default_button and get_cancel_button methods.

Example:
// Make a simple form with a few buttons.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
form.set_button_attributes(ok,  true,  false);
form.set_button_attributes(cancel,  false,  true);
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will set the checked or unchecked state of a checkbox.

bool set_checkbox_mark(int control_id, bool checked)

Parameters:
control_id
The ID of the control you wish to change.
checked
A boolean value indicating whether the control should be checked. A value of true means the checkbox will be checked, while a
value of false means the checkbox will not be checked.

Return value:
true on success, false on failure.

Remarks:
None.

Example:
// Make a simple form with a checkbox that changes state after its creation.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  checkbox=form.create_checkbox("&Change  me",  true);
form.set_checkbox_mark(checkbox,  false);
int  ok=form.create_button("OK");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will set the position of the cursor in a listbox.

bool set_list_position(int control_id, int new_position)

Parameters:
control_id
The control ID of the listbox.
new_position
The position in the listbox to highlight. -1 indicates that no item is to be highlighted.

Return value:
true on success, false on failure.

Remarks:
This is useful for automatically filling in information into a form.

Please note that the position does not need to be manually set when monitoring for input, since the audio_form class changes the
position automatically when different items are selected.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
form.set_list_position(list,  2);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
alert("Information",  "The  listbox  cursor  is  at  position  "+form.get_list_position(list));
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will set the various outputs for the form.

bool set_output_mode(int speech_output, bool beeping_progress_bar=false)

Parameters:
speech_output
The type of speech output to use.
beeping_progress_bars
An optional parameter indicating whether progress bars should speak percentages or indicate them with beeps.

Return value:
true on success, false on failure.

Remarks:
The speech_output parameter can be 0 to use Sapi, or it can correspond to one of the four screen readers that are currently
supported in BGT.

Example:
// Make a simple form with a few buttons and a text field, and get Jaws to read the form.

#include  "form.bgt"

audio_form  form;

void  main()
{
install_keyhook();
form.set_output_mode(JAWS);
form.create_window("Example  Form",  true);
int  test=form.create_input_box("Test",  "test",  "",  0,  false,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
uninstall_keyhook();
exit();
}
if(form.is_pressed(cancel))
{
uninstall_keyhook();
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
uninstall_keyhook();
exit();
}
}
}

audio_form object

This method will set the overwrite mode of an input box.

bool set_overwrite_mode(int control_id, bool overwrite)

Parameters:
control_id
The ID of the control you wish to change.
overwrite
A boolean value indicating whether the control should have overwrite enabled or disabled.

Return value:
true on success, false on failure.

Remarks:
If overwrite is enabled, new text that is entered into the control will overwrite old highlighted text rather than adding to it.

Example:
// Make a form with a few input boxes, and change the overwrite mode with one of them.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  edit=form.create_input_box("Edit  me.");
int  overwrite=form.create_input_box("Overwrite  me.",  "Overwrite  me.");
form.set_overwrite_mode(overwrite,  true);
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will set the value of a progress bar.

bool set_progress(int control_id, int value)

Parameters:
control_id
The control ID to retrieve the progress from.
value
The new percentage of the progress bar.

Return value:
true on success, false on failure.

Remarks:
This method will only function on progress bars.

As a progress bar deals only with percentages, valid values are 0 to 100. Any values above or below are ignored and an error is
flagged.

Example:
// Make a simple form with a few buttons and a dummy progress bar.

#include  "form.bgt"

audio_form  form;
int  dummy_counter;

void  main()
{
dummy_counter=0;
form.create_window("Example  Form",  true);
int  dummy=form.create_progress_bar("&Copying",  0,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
dummy_counter++;
if(dummy_counter>=250)
{
form.set_progress(dummy,  form.get_progress(dummy)+1);
dummy_counter=0;
}
if(form.is_pressed(ok))
{
alert("Information",  "The  current  value  of  the  progress  bar  is  "+form.get_progress(dummy));
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will set various speech settings for controls.

bool set_speech_verbosity_options(int control_id, string highlight_selection, string highlight_unselection, string deletion, string percentage,
int keyboard_echo)

Parameters:
control_id
The ID of the control you wish to change.
highlight_selection
The text that is to be spoken when text is highlighted.
highlight_unselection
The text to be spoken when text is no longer highlighted.
deletion
The text that is to be spoken when text is deleted.
percentage
The text to be spoken after the value of the progress bar is read.
keyboard_echo
One of the input box speech flags (see constants).

Return value:
true on success, false on failure.

Remarks:
None.

Example:
// Make a simple form with a few buttons and a text field.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  test=form.create_input_box("Test",  "test",  "",  0,  false,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
form.set_speech_verbosity_options(test,  "Selected",  "Unselected",  "Deleted",  "%",  textflag_characters);
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will set the enabled and visible states of a control.

bool set_state(int control_id, bool enabled, bool visible)

Parameters:
control_id
The ID of the control you wish to change.
enabled
A boolean value indicating whether the control should be enabled.
visible
A boolean value indicating whether the control should be visible.

Return value:
true on success, false on failure.

Remarks:
The enabled/disabled flag refers to whether the control may be manipulated in any way. If set to disabled, the audio feedback
will announce this and the control is unusable.
The visibility flag refers to whether the control is accessible whilst tabbing through the dialog. If it is set to invisible, the form's
cursor will skip the control but the control may still be used and/or modified by the script.

Example:
// Make a simple form with a few buttons and a listbox.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  list=form.create_list("&People",  0);
form.add_list_item(list,  "Joseph",  -1);
form.add_list_item(list,  "Samuel",  -1);
form.add_list_item(list,  "Percival",  1);
form.add_list_item(list,  "William",  -1);
form.add_list_item(list,  "Edward",  0);
int  ok=form.create_button("OK");
form.set_state(ok,  false,  false);
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method will change the text of a given control.

bool set_text(int control_id, string text)

Parameters:
control_id
The ID of the control which is to receive the new text.
text
The new text for the control.

Return value:
true on success, false on failure.

Remarks:
This method will only function on text fields and status bars.

Example:
// Make a simple form with a few buttons and a text field.

#include  "form.bgt"

audio_form  form;

void  main()
{
form.create_window("Example  Form",  true);
int  test=form.create_input_box("Test",  "",  "",  0,  false,  false);
int  ok=form.create_button("OK");
int  cancel=form.create_button("Eξt");
while(true)
{
form.monitor();
wait(5);
if(form.is_pressed(ok))
{
form.set_text(test,  "diddy  daddy  dumbo");
alert("Information",  "The  text  entered  in  the  text  field  is  "+form.get_text(test));
exit();
}
if(form.is_pressed(cancel))
{
exit();
}
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
}
}

audio_form object

This method allows you to specify an existing tts_voice object to be used for any Sapi 5 spoken output in the form.

bool set_tts_object(tts_voice@ handle)

Parameters:
handle
A handle to an existing tts_voice object that is to be used for all subsequent Sapi 5 output.

Return value:
true on success, false on failure.

Remarks:
This method is useful if you wish to preserve any settings such as rate, volume etc of an existing tts_voice object. The object that
you specify in this function will be used for all subsequent text strings spoken by Sapi 5. If you do not call this method and still
use Sapi output, an internal tts_voice object with default settings will be used instead.

Example:
// Make a simple form with redirected Sapi.

#include  "form.bgt"

audio_form  form;
tts_voice  my_voice;

void  main()
{
my_voice.rate=120;
form.set_tts_object(my_voice);
form.create_window("Test",  true);
while(true)
{
form.monitor();
wait(5);
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
my_voice.speak_interrupt_wait("Exiting.");
exit();
}
}
}

audio_form object

bool speak_control_attributes_separately
This property determines whether control attributes are spoken separately. If set to true, there will be a pause between the
announcement of the control's caption and its type. Otherwise the control caption and type are spoken together.

bool active
This property indicates whether the form is currently active. This is usually the case between calls to the create and destroy
methods.

dynamic_menu object

The dynamic_menu object is stored in dynamic_menu.bgt in the BGT include directory, and is used to create an audio or
Microsoft Sapi 5 text to speech (TTS) based menu.

dynamic_menu()

Parameters:
None.

Remarks:
The dynamic_menu object allows you to create a menu by adding options and then starting the menu and letting the user choose
one of the options you have added. You can then act based on the information that the menu passes back to you.

It is perfectly legal to have both audio and tts items in one and the same menu.

Example:
//Make  a  simple  menu.

#include  "dynamic_menu.bgt"

void  main()
{
dynamic_menu  my_menu;
int  menu_result;
my_menu.allow_escape=true;
my_menu.wrap=true;
my_menu.add_item("start_game.wav");
my_menu.add_item_tts("test  speakers");
my_menu.add_item("exit.wav");
menu_result=my_menu.run("choose_an_option.wav",  false);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

dynamic_menu object

The dynamic_menu object is stored in dynamic_menu.bgt in the BGT include directory, and is used to create an audio or
Microsoft Sapi 5 text to speech (TTS) based menu.

dynamic_menu()

Parameters:
None.

Remarks:
The dynamic_menu object allows you to create a menu by adding options and then starting the menu and letting the user choose
one of the options you have added. You can then act based on the information that the menu passes back to you.

It is perfectly legal to have both audio and tts items in one and the same menu.

Example:
//Make  a  simple  menu.

#include  "dynamic_menu.bgt"

void  main()
{
dynamic_menu  my_menu;
int  menu_result;
my_menu.allow_escape=true;
my_menu.wrap=true;
my_menu.add_item("start_game.wav");
my_menu.add_item_tts("test  speakers");
my_menu.add_item("exit.wav");
menu_result=my_menu.run("choose_an_option.wav",  false);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

dynamic_menu object

This method will add an item to a menu using audio as its output.

int add_item(string filename, string name="")

Parameters:
filename
The file to add. If the sound is stored on disk this can be either an absolute path, a relative path or the file name on its own.
Absolute paths are not applicable however, if the sound is being added from a pack file (see remarks).
name
An optional name intended to be used for lookups (see remarks).

Return value:
The position in the menu of the new item (starting at 1) on success, -1 on failure.

Remarks:
The location in which the engine searches for the file specified in the filename parameter is determined by a call to the
"set_sound_storage" function. By default it will look on the users hard drive and in the directory where the program is stored,
unless an absolute path is specified such as "C:\\Windows\\media\\ding.wav". If the engine has been set up to look inside a pack
file then only a file name such as "creature.ogg" or a relative path such as "sounds\\monster.ogg" can be specified.

Note that both \ and / can be used to specify paths.

Item names are handy when you do not know in what order items may appear, or indeed if certain items will appear at all. For a
more thorough description of item names and their uses, please see the get_item_name method chapter.

Example:
// Make a simple menu.

#include  "dynamic_menu.bgt"

void  main()
{
dynamic_menu  my_menu;
int  menu_result;
my_menu.allow_escape=true;
my_menu.wrap=false;
my_menu.add_item("start_game.wav");
my_menu.add_item("test_speakers.wav");
my_menu.add_item("exit.wav");
menu_result=my_menu.run("choose_an_option.wav",  false);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)

{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

dynamic_menu object

This method will add an item to a menu using Microsoft Sapi 5.1 or a screen reader as its output.

int add_item_tts(string text_to_speak, string name="")

Parameters:
text_to_speak
The text to add as an option.
name
An optional name intended to be used for lookups (see remarks).

Return value:
The position in the menu of the new item (starting at 1) on success, -1 on failure.

Remarks:
By default, a temporary tts-voice object will be created in order to speak any menu items directed to Sapi if the speech mode is
0. To change this, you may specify an already existing tts_voice object to be used wherever Sapi output is desired, by calling the
set_tts_object method.

Item names are handy when you do not know in what order items may appear, or indeed if certain items will appear at all. For a
more thorough description of item names and their uses, please see the get_item_name method chapter.

Example:
// Make a simple menu using Microsoft Sapi 5.

#include  "dynamic_menu.bgt"

void  main()
{
dynamic_menu  my_menu;
int  menu_result;
my_menu.add_item_tts("Start  game.");
my_menu.add_item_tts("Test  speakers.");
my_menu.add_item_tts("Exit.");
menu_result=my_menu.run("choose  an  option.",  true);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

dynamic_menu object

This method will tell you how many items have been added to the menu.

double get_item_count()

Parameters:
None.

Return value:
A double indicating the number of items currently added to the menu.

Remarks:
None.

Example:
// Make a simple menu and retrieve the number of items.

#include  "dynamic_menu.bgt"

void  main()
{
dynamic_menu  test;
test.add_item("start_game.wav");
test.add_item("test_speakers.wav");
test.add_item("exit.wav");
alert("Menu  example",  test.get_item_count()+"  items  are  currently  stored  in  the  menu.");
}

dynamic_menu object

This method will retrieve the name for an item.

string get_item_name(int item

Parameters:
item
The item to retrieve the data from.

Return value:
The item name on success, or a blank string on failure or if no data is set.

Remarks:
Item names are handy when you do not know in what order items may appear, or indeed if certain items will appear at all. For instance, if the game is not
registered you may want an option to register the game, and remove this option if the user has purchased a copy. If you do this using the numeric return value
from the run or run_extended method, you will need to calculate what number all of the other items will have, based on whether the register item is there or not.
This can quickly become unmanageable if you have multiple items that are dynamic in this fashion.

To circumvent this problem, it is a good idea to assign a unique name to each item. Then you are able to call this method with the return value from the run or the
run_extended method as the item number, and retrieve the name that you assigned to it when you added it to the menu. Therefore, if you assign the name "exit"
to a given item, it does not matter in what order the items appear; you can always find exit based on its name.

Note that the item name is not in any way related to the sound filename, or the text that is to be spoken by a given synthetic voice. The item name is purely
intended for lookups in the code, and are never shown to the user.

Example:
// Make a simple menu where all the items use names.

#include  "dynamic_menu.bgt"

void  main()
{
dynamic_menu  my_menu;
int  menu_result;
my_menu.allow_escape=true;
my_menu.wrap=false;
my_menu.add_item("start_game.wav",  "start");
my_menu.add_item("test_speakers.wav",  "test");
my_menu.add_item("exit.wav",  "exit");
menu_result=my_menu.run("choose_an_option.wav",  false);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}

//  Now  we  get  the  name  for  the  item,  and  figure  out  what  to  do  based  on  that.
// Note that, to demonstrate the usefulness of this, we don't actually check for the items in the order they were added!
string  name=my_menu.get_item_name(menu_result);
if(name=="test")
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(name=="exit")
{

alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
if(name=="start")
{
alert("Option",  "Option  selected  was  start  game.");
}
}

dynamic_menu object

This method returns the current position of the menu cursor.

double get_position()

Parameters:
None.

Return value:
A double indicating the position of the menu cursor on success, or -1 on failure.

Remarks:
This method may only be used while the menu is active. Thus, it can only be used from within a callback function.

Example:
// Make a simple menu and use a callback. The callback will check the current position with a variable and play a sound if the position has changed.

#include  "dynamic_menu.bgt"

sound  movement;
dynamic_menu  menu;
int  menu_position;

void  main()
{
int  menu_result;
menu.allow_escape=true;
menu.wrap=false;
menu.set_callback(check_menu_events,  "");
menu.add_item_tts("Start  game.");
menu.add_item_tts("Test  speakers.");
menu.add_item_tts("Exit.");
menu_result=menu.run("Please  choose  an  option",  true);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

int  check_menu_events(dynamic_menu@  game_menu,  string  data)
{
if(game_menu.get_position()!=menu_position)
{
movement.load("menumove.wav");

movement.play_wait();
movement.close();
menu_position=game_menu.get_position();
}
return  0;
}

dynamic_menu object

This method will check to see if the menu is running.

bool is_running()

Parameters:
None.

Return value:
true on success, false on failure.

Remarks:
This is especially useful in menu callbacks if you only want to perform a certain action while the menu is running.

Example:
// Make a simple menu and use a callback. The callback will check the current position with a variable and play a sound if the position has changed. The callback will additionally check if the menu is running.

#include  "dynamic_menu.bgt"

sound  movement;
dynamic_menu  menu;
int  menu_position;

void  main()
{
int  menu_result;
menu.allow_escape=true;
menu.wrap=false;
menu.set_callback(check_menu_events,  "");
menu.add_item_tts("Start  game.");
menu.add_item_tts("Test  speakers.");
menu.add_item_tts("Exit.");
menu_result=menu.run("Please  choose  an  option",  true);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

int  check_menu_events(dynamic_menu@  game_menu,  string  data)
{
if(!game_menu.is_running)
{
return  -1;

}
if(game_menu.get_position()!=menu_position)
{
movement.load("menumove.wav");
movement.play_wait();
movement.close();
menu_position=game_menu.get_position();
}
return  0;
}

dynamic_menu object

This method will reset the menu, removing previously added items and, if required, the options previously set.

bool reset(bool completely)

Parameters:
completely
A boolean value specifying whether the options should be reset in addition to the menu items.

Return value:
true on success, false on failure.

Remarks:
This is useful if you need to rebuild a new menu with the object, for example to make a submenu.

Example:
// Make a simple menu.

#include  "dynamic_menu.bgt"

dynamic_menu  my_menu;

void  main()
{
main_menu();
}

void  main_menu()
{
my_menu.reset(true);
my_menu.allow_escape=true;
my_menu.wrap=false;
my_menu.add_item("start_game.wav");
my_menu.add_item("test_speakers.wav");
my_menu.add_item("exit.wav");
menu_result=my_menu.run("choose_an_option.wav",  false);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
difficulty_menu();
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

void  difficulty_menu()

{
my_menu.reset(false);
my_menu.add_item("easy.wav");
my_menu.add_item("medium.wav");
my_menu.add_item("hard.wav");
my_menu.add_item("return_to_main_menu.wav");
menu_result=my_menu.run("choose_an_option.wav",  false);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  easy.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  medium.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  hard.");
}
if(menu_result==4)
{
alert("Option",  "Option  selected  was  return  to  main  menu.");
main_menu();
}
}

dynamic_menu object

This method will run the menu with the options that you have specified.

double run(string intro, bool is_intro_tts)

Parameters:
intro
The sound file to play or the intro text that will be spoken using Microsoft Sapi before the options are revealed. If the sound is
stored on disk this can be either an absolute path, a relative path or the file name on its own. Absolute paths are not applicable
however, if the sound is being loaded from a pack file (see remarks).
is_intro_tts
A boolean specifying whether the intro string is a filename or text to be spoken. If set to true, the string will be spoken using
Microsoft Sapi 5. If set to false, the string is assumed to be a filename that the menu class will load and play.

Return value:
A double value indicating which option was selected in the menu, or -1 if there was an error. If the allow_escape property is set
to true, the method will return 0 if the Escape key was pressed.

Remarks:
If the intro_tts parameter is set to false, the location in which the engine searches for the file specified in the intro parameter is
determined by a call to the "set_sound_storage" function. By default it will look on the users hard drive and in the directory
where the program is stored, unless an absolute path is specified such as "C:\\Windows\\media\\ding.wav". If the engine has been
set up to look inside a pack file then only a file name such as "creature.ogg" or a relative path such as "sounds\\monster.ogg" can
be specified.

Note that both \ and / can be used to specify paths.

If an empty string is given for the intro parameter, no menu intro will be heard.

Example:
// Make a simple menu.

#include  "dynamic_menu.bgt"

void  main()
{
dynamic_menu  my_menu;
int  menu_result;
my_menu.allow_escape=true;
my_menu.wrap=false;
my_menu.add_item("start_game.wav");
my_menu.add_item("test_speakers.wav");
my_menu.add_item("exit.wav");
menu_result=my_menu.run("choose_an_option.wav",  false);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)

{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

dynamic_menu object

This method will run the menu with the options that you have specified, and optionally start at an arbitrary position in the menu as
well as speak the current item at the end of the intro message.

double run_extended(string intro, bool is_intro_tts, double start_position, bool speak_initial_item)

Parameters:
intro
The sound file to play or the intro text that will be spoken using Microsoft Sapi before the options are revealed. If the sound is
stored on disk this can be either an absolute path, a relative path or the file name on its own. Absolute paths are not applicable
however, if the sound is being loaded from a pack file (see remarks).
is_intro_tts
A boolean specifying whether the intro string is a filename or text to be spoken. If set to true, the string will be spoken using
Microsoft Sapi 5. If set to false, the string is assumed to be a filename that the menu class will load and play.
double start_position
The position at which the menu cursor should initially be placed, starting from 1.
bool speak_initial_item
A boolean specifying whether the menu item that the cursor initially is placed at should be spoken after the intro message has
finished.

Return value:
A double value indicating which option was selected in the menu, or -1 if there was an error. If the allow_escape property is set
to true, the method will return 0 if the Escape key was pressed.

Remarks:
If the intro_tts parameter is set to false, the location in which the engine searches for the file specified in the intro parameter is
determined by a call to the "set_sound_storage" function. By default it will look on the users hard drive and in the directory
where the program is stored, unless an absolute path is specified such as "C:\\Windows\\media\\ding.wav". If the engine has been
set up to look inside a pack file then only a file name such as "creature.ogg" or a relative path such as "sounds\\monster.ogg" can
be specified.

Note that both \ and / can be used to specify paths.

If an empty string is given for the intro parameter, no menu intro will be heard.

Example:
// Make a simple menu, starting at the first item and having it automatically announced.

#include  "dynamic_menu.bgt"

void  main()
{
dynamic_menu  my_menu;
int  menu_result;
my_menu.allow_escape=true;
my_menu.wrap=false;
my_menu.add_item("start_game.wav");
my_menu.add_item("test_speakers.wav");
my_menu.add_item("exit.wav");
menu_result=my_menu.run_extended("choose_an_option.wav",  false,  1,  true);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");

exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

dynamic_menu object

This method allows you to specify a user defined function that can be used continually from within the menu loop.

bool set_callback(menu_callback@ callback, string user_data)

Parameters:
callback
The handle to a callback function that will be called continually within the menu loop.
user_data
A string containing any custom data that will be sent to the callback.

Return value:
true on success, false on failure.

Remarks:
The callback function should contain two parameters; the handle to a dynamic_menu object and a string for the custom user data. It should return an integer, which should be 0 if you wish the menu
to continue executing. If any other value than 0 is returned, the menu will stop and the return value from the callback will be returned as the menu result.

This method is useful for performing various operations from within the menu, such as poling for network events, allowing the user to change the music volume, or playing sounds when the menu
position has changed.

Example:
// Make a simple menu and use a callback. The callback will check the current position with a variable and play a sound if the position has changed.

#include  "dynamic_menu.bgt"

sound  movement;
dynamic_menu  menu;
int  menu_position;

void  main()
{
int  menu_result;
menu.allow_escape=true;
menu.wrap=false;
menu.set_callback(check_menu_events,  "");
menu.add_item_tts("Start  game.");
menu.add_item_tts("Test  speakers.");
menu.add_item_tts("Exit.");
menu_result=menu.run("Please  choose  an  option",  true);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");

exit();
}
}

int  check_menu_events(dynamic_menu@  game_menu,  string  data)
{
if(game_menu.get_position()!=menu_position)
{
movement.load("menumove.wav");
movement.play_wait();
movement.close();
menu_position=game_menu.get_position();
}
return  0;
}

dynamic_menu object

This method will set the speech output for the menu.

bool set_speech_mode(int speech_output)

Parameters:
speech_output
The type of speech output to use.

Return value:
true on success, false on failure.

Remarks:
The speech_output parameter can be 0 to use Sapi, or it can correspond to one of the four screen readers that are currently
supported in BGT.

Example:
// Make a simple menu using the Jaws screen reader.

#include  "dynamic_menu.bgt"

void  main()
{
dynamic_menu  my_menu;
int  menu_result;
my_menu.add_item_tts("Start  game.");
my_menu.add_item_tts("Test  speakers.");
my_menu.add_item_tts("Exit.");
install_keyhook();
my_menu.set_speech_mode(JAWS);
menu_result=my_menu.run("choose  an  option.",  true);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

dynamic_menu object

This method allows you to specify an existing sound object to be used for any auditory feedback in the menu.

bool set_sound_object(sound@ handle)

Parameters:
handle
A handle to an existing sound object that is to be used for all subsequent sound output.

Return value:
true on success, false on failure.

Remarks:
The object that you specify in this function will be used for all subsequent auditory feedback provided to the user in the menu. If
you do not call this method and still use sound output, an internal sound object will be used instead.

Example:
//  Make  a  simple  menu  with  redirected  sound  playback.

#include  "dynamic_menu.bgt"

void  main()
{
sound  my_sound;
dynamic_menu  my_menu;
int  menu_result;
my_menu.allow_escape=true;
my_menu.wrap=false;
my_menu.set_sound_object(my_sound);
my_menu.add_item("start_game.wav");
my_menu.add_item("test_speakers.wav");
my_menu.add_item("exit.wav");
menu_result=my_menu.run("choose_an_option.wav",  false);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

dynamic_menu object

This method allows you to specify an existing tts_voice object to be used for any Sapi 5 spoken output in the menu.

bool set_tts_object(tts_voice@ handle)

Parameters:
handle
A handle to an existing tts_voice object that is to be used for all subsequent Sapi 5 output.

Return value:
true on success, false on failure.

Remarks:
This method is useful when you are using Sapi 5 spoken output in the menu and wish to preserve any settings such as rate,
volume etc of the voice. The object that you specify in this function will be used for all subsequent text strings spoken by Sapi 5.
If you do not call this method and still use Sapi output, an internal tts_voice object with the default settings will be used instead.

Note that even if you specify an explicit tts_voice object with this method, Sapi will not be used if the speech mode is not 0.

Example:
// Make a simple menu with redirected Sapi.

#include  "dynamic_menu.bgt"

void  main()
{
tts_voice  my_voice;
my_voice.rate=120;
dynamic_menu  my_menu;
int  menu_result;
my_menu.allow_escape=true;
my_menu.wrap=false;
my_menu.set_tts_object(my_voice);
my_menu.add_item_tts("Start  game.");
my_menu.add_item("Test  speakers.");
my_menu.add_item_tts("Exit.");
menu_result=my_menu.run("choose_an_option.wav",  false);
if(menu_result==-1)
{
alert("Error",  "There  was  an  error  loading  the  menu.");
exit();
}
if(menu_result==0)
{
alert("Option",  "Escape  was  pressed.  Exiting.");
exit();
}
if(menu_result==1)
{
alert("Option",  "Option  selected  was  start  game.");
}
if(menu_result==2)
{
alert("Option",  "Option  selected  was  test  speakers.");
}
if(menu_result==3)
{
alert("Option",  "Option  selected  was  exit.  Exiting.");
exit();
}
}

dynamic_menu object

bool wrap
This property indicates whether the menu will wrap. If set to true, the menu will cycle back to the first option if an attempt is
made to move past the final option, and vice versa.

bool allow_escape
This property indicates whether the user is allowed to press the Escape key. If set to true, the menu will return if either the Enter
key is pressed or the Escape key. Otherwise only enter is monitored.

bool enable_home_and_end
This property indicates whether the user should be allowed to press home and end to jump to the start or the end of the menu
respectively.

logger object

The logger object is contained in logger.bgt in the BGT include directory, and is used to write log files with relative ease.

logger()

Parameters:
None.

Remarks:
The logger class is useful for writing trace logs for debugging your game, or recording game statistics.

Example:
//Write a few log entries.

#include  "logger.bgt"

logger  test;

void  main()
{
test.header_text="Hello.  This  is  the  start  of  a  log.";
test.footer_text="This  is  the  end  of  the  log.  Goodbye.";
test.date_all_entries=true;
test.overwrite_file=false;
test.add_entry("Entry1.");
wait(2000);
test.add_entry("Entry2.");
alert("Thanks.",  "Log  was  written  successfully.");
test.add_entry("Logging  successful.");
test.write("test.log",  true);
}

logger object

The logger object is contained in logger.bgt in the BGT include directory, and is used to write log files with relative ease.

logger()

Parameters:
None.

Remarks:
The logger class is useful for writing trace logs for debugging your game, or recording game statistics.

Example:
//Write a few log entries.

#include  "logger.bgt"

logger  test;

void  main()
{
test.header_text="Hello.  This  is  the  start  of  a  log.";
test.footer_text="This  is  the  end  of  the  log.  Goodbye.";
test.date_all_entries=true;
test.overwrite_file=false;
test.add_entry("Entry1.");
wait(2000);
test.add_entry("Entry2.");
alert("Thanks.",  "Log  was  written  successfully.");
test.add_entry("Logging  successful.");
test.write("test.log",  true);
}

logger object

This method will add an entry to a log.

void add_entry(string the_entry)

Parameters:
the_entry
The text to add as an entry.

Return value:
None.

Remarks:
If the date_all_entries property is set to true, it will use the combination date_format+": "+the_entry to add the text, replacing all
the date controllers into actual date and time components.

Example:
//add some text to a log.

#include  "logger.bgt"

logger  test;

void  main()
{
test.add_entry("text1");
test.write("test.log",  true);
}

logger object

This method will reset the data that has been accumulated internally for the log.

void reset()

Parameters:
None.

Return value:
None.

Remarks:
This operation is irreversible.

Example:
//Add some text to the log, reset it, then add some more.

#include  "logger.bgt"

logger  test;

void  main()
{
test.add_entry("text1");
test.add_entry("text2");
test.add_entry("text3");
test.reset();
test.add_entry("text4");
test.add_entry("text5");
test.add_entry("text6");
test.add_entry("text7");
test.write("test.log",  true);
}

logger object

This method will write the data accumulated internally for the log to a text file.

bool write(string filename, bool reset_text)

Parameters:
filename
The name of the file to write to.
reset_text
A boolean value specifying whether the log should be reset. If set to false, the text will remain in the log until you manually call the
reset method.

Return value:
true on success, false on failure.

Remarks:
If an error occurs, no file will be written to and the text will remain in the log, regardless of the value of the reset_text parameter.

If the always_include_header property is set to true and the header_text property contains some text, the header will be written
to the log every time the write method is called. Otherwise the header will only be written if there is no text already in the
specified file.

The footer will only be included in the log if the reset_text parameter is set to true.

Example:
//add some text to a log.

#include  "logger.bgt"

log  test;

void  main()
{
test.add_entry("text1");
test.write("test.log",  true);
}

logger object

string header_text
This will be the string that will be prepended to any log file. This can be useful, for example, if you need to log when the program
started. Default is blank.

string footer_text
This will be the string that will be appended to any log file. This can be useful if you need to log when the program ended. Default
is blank.

string date_format
This specifies the date and time format that the log will use. This is fully customisable, and can be any text, using the following
combinations as date and time controllers enclosed in < and > signs (less than/greater than signs).

d or D=the date (1 to 31).
dd or DD=date (01 to 31).
w=weekday (1 to 7).
W=weekday (monday to sunday).
m=month (1 to 12).
mm or MM=month (01 to 12).
M=month (January to December).
y=year in 4-digit format.
Y=year in 2-digit format.
h=hour (1 to 12).
hh=hour (01 to 12).
H=hour (0 to 23).
HH=hour (00 to 23).
n or N=minute (0 to 59).
nn or NN=minute (00 to 59).
s or S=second (0 to 59).
ss or SS=second (00 to 59).
tt=am/pm.
TT=AM/PM.

Default value is "<dd>/<mm>/<y>, <hh>:<nn>:<ss> <tt>".

bool date_all_entries
This specifies whether all logged entries should be dated. If set to true, all entries will be dated. If set to false, no entries are
dated unless you manually enter the date. Default value is false.

bool overwrite_file
This specifies whether the log file will be appended to or overwritten. Set this to true to overwrite the file, or false to append.
Default value is false.

bool always_include_header
Specifies whether the header should always be included in the log. This property is ignored if the header_text property is blank.
Default is false.

number_speaker object

The number_speaker object is contained in number_speaker.bgt in the BGT include directory, and is used to assist in speaking
numbers as naturally and smoothly as possible.

number_speaker()

Parameters:
None.

Remarks:
The number_speaker class will look for a sound file that matches the words in the number as closely as possible, separating the
words with an underline. So if the number is 56, it will first look to see if there is a file called "fifty_six.wav". If this is not found,
it'll look to see if there is a file called "fifty.wav" and another called "six.wav" and use those instead. In short this means that you
are able to record as many numbers as you wish to make the spoken output sound as flawless as possible, or you may record
only the necessary ones which is to say "zero.wav" to "nineteen.wav", "twenty.wav" to "ninety.wav" and then optionally
"hundred.wav", "thousand.wav" and "million.wav". You may also make a file called "minus.wav" if you wish to distinguish
between positive and negative numbers.

Example:
//speak  a  number  using  speak_wait.

#include  "number_speaker.bgt"

void  main()
{
number_speaker  test;
test.speak_wait(1056);
}

number_speaker object

The number_speaker object is contained in number_speaker.bgt in the BGT include directory, and is used to assist in speaking
numbers as naturally and smoothly as possible.

number_speaker()

Parameters:
None.

Remarks:
The number_speaker class will look for a sound file that matches the words in the number as closely as possible, separating the
words with an underline. So if the number is 56, it will first look to see if there is a file called "fifty_six.wav". If this is not found,
it'll look to see if there is a file called "fifty.wav" and another called "six.wav" and use those instead. In short this means that you
are able to record as many numbers as you wish to make the spoken output sound as flawless as possible, or you may record
only the necessary ones which is to say "zero.wav" to "nineteen.wav", "twenty.wav" to "ninety.wav" and then optionally
"hundred.wav", "thousand.wav" and "million.wav". You may also make a file called "minus.wav" if you wish to distinguish
between positive and negative numbers.

Example:
//speak  a  number  using  speak_wait.

#include  "number_speaker.bgt"

void  main()
{
number_speaker  test;
test.speak_wait(1056);
}

number_speaker object

This method allows you to specify an existing sound object to be used for any auditory feedback in the number speaker.

bool set_sound_object(sound@ handle)

Parameters:
handle
A handle to an existing sound object that is to be used for all subsequent sound output.

Return value:
true on success, false on failure.

Remarks:
The object that you specify in this function will be used for all subsequent auditory feedback provided to the user in the number
speaker. If you do not call this method and still use sound output, an internal sound object will be used instead.

Example:
//  Speak  a  number  with  redirected  sound  playback.

#include  "number_speaker.bgt"

void  main()
{
sound  my_sound;
number_speaker  number;
number.set_sound_object(my_sound);
number.speak_wait(500);
}

number_speaker object

This method will speak a number from -999999999 to 999999999.

int speak(double the_number)

Parameters:
the_number
The number to speak.

Return value:
0 on success, -1 on failure.

Remarks:
The speak method will look for sounds in accordance with the values stored in the prepend and append properties, and ascertain
the best sound files to load by performing a number of searches based on the number at hand.

This function will return immediately when called, allowing the script to do other things while the number is being spoken. It is
therefore necessary to call the speak_next method continuously to ensure that the numbers are spoken as smoothly and fluently
as possible.

Example:
// Speak a number using speak and speak_next.

#include  "number_speaker.bgt"

void  main()
{
number_speaker  test;
test.prepend="numbers/";
test.speak(1056);
while(test.speak_next()==1)
{
wait(5);
}
}

number_speaker object

This method is used to check the status of a number that is or has been spoken, and where necessary, starts playing the next one.

int speak_next()

Parameters:
None.

Return value:
0 if there are no more files to be played, 1 if there are more files yet to be played or if the last file has not yet finished playing, or
-1 if an error occurs.

Remarks:
The speak_next method works in conjunction with the speak method, and is to be used continuously while the number is being
spoken.

This method works not only as a status indicator, but also as a trigger, checking and playing the numbers where necessary.
Therefore, to ensure that the numbers are spoken as smoothly and as fluently as possible, it is essential that this check is
performed at regular intervals, I.E. every 5 or so milliseconds.

Example:
// Speak a number using speak and speak_next.

#include  "number_speaker.bgt"

void  main()
{
number_speaker  test;
test.prepend="numbers/";
test.speak(1056);
while(test.speak_next()==1)
{
wait(5);
}
}

number_speaker object

This method will speak a number between -999999999 and 999999999, and wait until the number is fully read before returning.

int speak_wait(int the_number)

Parameters:
the_number
The number to speak.

Return value:
0 on success, -1 on failure.

Remarks:
The speak_wait method will look for sounds in accordance with the values stored in the prepend and append properties, and
ascertain the best sound files to load by performing a number of searches based on the number at hand.

The speak_wait function waits until the number is fully spoken before returning. This means that script execution pauses
throughout the entire reading. If you wish to avoid this, use the speak and speak_next methods.

Example:
//  Speak  a  number  using  speak_wait.

#include  "number_speaker.bgt"

void  main()
{
number_speaker  test;
test.speak_wait(1056);
}

number_speaker object

This method stops any number that is currently being spoken.

void stop()

Parameters:
None.

Return value:
None.

Remarks:
The speak_next method works in conjunction with the speak method, and is to be used continuously while the number is being
spoken.

This method works not only as a status indicator, but also as a trigger, checking and playing the numbers where necessary.
Therefore, to ensure that the numbers are spoken as smoothly and as fluently as possible, it is essential that this check is
performed at regular intervals, I.E. every 5 or so milliseconds.

Example:
// Speak a number using speak and speak_next, stopping it prematurely if the user presses space.

#include  "number_speaker.bgt"

void  main()
{
number_speaker  test;
test.prepend="numbers/";
test.speak(1056);
while(test.speak_next()==1)
{
if(key_pressed(KEY_SPACE))
test.stop();
wait(5);
}
}

number_speaker object

string prepend
This will be the string that will be prepended to any number file that is loaded. This is useful, for example, if your numbers are
stored in a separate directory, or have a prefix on the filenames.

string append
This will be the string that will be appended to any number file that is loaded. The default is .wav, so it is only necessary to
change this if your numbers have a different extension.

bool include_and
This boolean decides whether the word "and" should be inserted in the appropriate places in the output. If set to true, you must
make a file called and.wav, unless the extention (e.g. the append property) is something other than .wav. The default value is
false.

sound_pool object

The sound_pool object is stored in sound_pool.bgt in the BGT include directory. It provides a convenient way of managing
sounds in an environment, both with 1 and 2 dimensions. It attempts to abstract the task of simulating auditory environments by
hiding some of the more intricate details in simpler method calls, and does not require the manipulation of any sound objects
directly.

sound_pool(int number_of_items)

Parameters:
number_of_items
An optional parameter specifying the number of sounds to allocate in the game world. If this parameter is not passed, the sound
pool will be given a default allocation of 100 sounds.

Remarks:
This object uses the functions found in sound_positioning.bgt to do the actual sound adjustments, so modifying these functions
will affect the entire system.

Please note that a non-looping sound source will be cleaned up once it has finished playing, in order to keep memory usage to an
absolute minimum at all times.

This object is very useful for playing sounds of a dynamic nature, for example when you don't know at the time of the game
design how many sounds you are ultimately going to need. When you call one of the play methods, the class will attempt to find a
free slot in which to play the sound. This can be used for anything from playing overlapping gunshots when the player fires
rapidly, to footsteps that need to play quickly without cutting out the previous one, to sounds of things that are occuring
dynamically in the environment, etc. Making good use of sound pools can significantly speed up your development time as well
as increase the over-all performance of your game.

Example:
//a  mini  sidescroller  using  the  sound_pool  class

#include  "sound_pool.bgt"

sound_pool  sound_environment;
timer  walk_timer;

int  player_position;

const  int  left=-1;
const  int  right=1;

const  int  boundary=200;

void  main()
{
show_game_window("sound_pool  example");
sound_environment.max_distance=70;
for(int  counter=1;  counter<15;  counter++)
{
sound_environment.play_1d("sounds/sound"+counter+".wav",  0,  random(0,  boundary),  true);
}
while(true)
{
check_input();
wait(5);
}
}

void  check_input()
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))

{
exit();
}
if(key_down(KEY_LEFT))
{
walk(left);
}
if(key_down(KEY_RIGHT))
{
walk(right);
}
}

void  walk(int  direction)
{
if((direction<=left)&&(player_position<=0))
{
return;
}
if((direction>=right)&&(player_position>=boundary))
{
return;
}
if(walk_timer.elapsed<350)
{
return;
}
walk_timer.restart();
player_position+=direction;
sound_environment.update_listener_1d(player_position);
}

sound_pool object

The sound_pool object is stored in sound_pool.bgt in the BGT include directory. It provides a convenient way of managing
sounds in an environment, both with 1 and 2 dimensions. It attempts to abstract the task of simulating auditory environments by
hiding some of the more intricate details in simpler method calls, and does not require the manipulation of any sound objects
directly.

sound_pool(int number_of_items)

Parameters:
number_of_items
An optional parameter specifying the number of sounds to allocate in the game world. If this parameter is not passed, the sound
pool will be given a default allocation of 100 sounds.

Remarks:
This object uses the functions found in sound_positioning.bgt to do the actual sound adjustments, so modifying these functions
will affect the entire system.

Please note that a non-looping sound source will be cleaned up once it has finished playing, in order to keep memory usage to an
absolute minimum at all times.

This object is very useful for playing sounds of a dynamic nature, for example when you don't know at the time of the game
design how many sounds you are ultimately going to need. When you call one of the play methods, the class will attempt to find a
free slot in which to play the sound. This can be used for anything from playing overlapping gunshots when the player fires
rapidly, to footsteps that need to play quickly without cutting out the previous one, to sounds of things that are occuring
dynamically in the environment, etc. Making good use of sound pools can significantly speed up your development time as well
as increase the over-all performance of your game.

Example:
//a  mini  sidescroller  using  the  sound_pool  class

#include  "sound_pool.bgt"

sound_pool  sound_environment;
timer  walk_timer;

int  player_position;

const  int  left=-1;
const  int  right=1;

const  int  boundary=200;

void  main()
{
show_game_window("sound_pool  example");
sound_environment.max_distance=70;
for(int  counter=1;  counter<15;  counter++)
{
sound_environment.play_1d("sounds/sound"+counter+".wav",  0,  random(0,  boundary),  true);
}
while(true)
{
check_input();
wait(5);
}
}

void  check_input()
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))

{
exit();
}
if(key_down(KEY_LEFT))
{
walk(left);
}
if(key_down(KEY_RIGHT))
{
walk(right);
}
}

void  walk(int  direction)
{
if((direction<=left)&&(player_position<=0))
{
return;
}
if((direction>=right)&&(player_position>=boundary))
{
return;
}
if(walk_timer.elapsed<350)
{
return;
}
walk_timer.restart();
player_position+=direction;
sound_environment.update_listener_1d(player_position);
}

sound_pool object

This method will destroy all of the sounds in the sound pool.

void destroy_all()

Parameters:
None.

Return value:
None.

Remarks:
This is useful if you want to return to the main menu, or reset to another level, etc.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;

void  main()
{
sounds.max_distance=70;
sounds.play_stationary("sounds/ambience.wav",  true);
wait(5000);
sounds.destroy_all();
exit();
}

sound_pool object

This method will destroy an active sound in the sound pool.

bool destroy_sound(int slot)

Parameters:
slot
The slot of the sound to destroy.

Return value:
true on success, false on failure.

Remarks:
This is useful if the character or object linked with the sound in question has died, for instance.

Please note: When dealing with sound slots, be sure that you set the persistent flag to true for all non-looping sounds when you
first create them. If you fail to do this, manipulating a sound in any way by use of its slot number can have unpredictable results.
This is because the sound pool automatically cleans up any sound that has finished playing and that is not set to be persistent,
with the result that the slot that was returned on creation is no longer invalid and may, in the worst case scenario, refer to a
completely different sound.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;

void  main()
{
sounds.max_distance=70;
int  slot=sounds.play_stationary("sounds/ambience.wav",  true);
wait(5000);
sounds.destroy_sound(slot);
}

sound_pool object

This method pauses all of the sounds in the sound pool.

void pause_all()

Parameters:
None.

Return value:
None.

Remarks:
To restart all the sounds, you must call the resume_all method or call the resume_sound method on each slot.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;

void  main()
{
sounds.max_distance=70;
sounds.play_stationary("sounds/ambience.wav",  true);
wait(5000);
sounds.pause_all();
wait(5000);
sounds.resume_all();
wait(5000);
}

sound_pool object

This method pauses an active sound.

bool pause_sound(int slot)

Parameters:
slot
The slot of the sound to pause.

Return value:
true on success, false on failure.

Remarks:
If the sound is already paused, you must call the resume_sound method to resume it.

Please note: When dealing with sound slots, be sure that you set the persistent flag to true for all non-looping sounds when you
first create them. If you fail to do this, manipulating a sound in any way by use of its slot number can have unpredictable results.
This is because the sound pool automatically cleans up any sound that has finished playing and that is not set to be persistent,
with the result that the slot that was returned on creation is no longer invalid and may, in the worst case scenario, refer to a
completely different sound.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;

void  main()
{
sounds.max_distance=70;
int  slot=sounds.play_stationary("sounds/ambience.wav",  true);
wait(5000);
sounds.pause_sound(slot);
wait(5000);
sounds.resume_sound(slot);
wait(5000);
}

sound_pool object

This method will play a sound that can move around according to the listener and source positions on a 1d grid.

int play_1d(string filename, int listener_x, int sound_x, bool looping, bool persistent=false)

Parameters:
filename
The filename to play.
listener_x
The position of the listener on the grid.
sound_x
The position of the source object on the grid.
looping
A boolean specifying whether the sound should loop or not.
persistent
An optional boolean specifying whether the sound should be persistent or not, meaning whether or not it should be automatically
cleaned up after playback has finished. If this argument is not given, false is the default which means that the sound does get
cleaned up.

Return value:
Returns the slot where the sound is stored in the pool, -1 on error, or -2 if the sound is out of earshot.

Remarks:
This method is useful for playing sounds specific to the game objects and characters, for example monsters, health boosters,
teleporters, etc.

A sound that is created with the persistent flag will not be cleaned up after it has finished playing, which otherwise is the default
behavior. Instead, an explicit call must be made to destroy the sound. This is useful for non-looping sounds that you wish to be
able to reliably manipulate with any of the functions that use slots to identify a particular sound. When set as persistent, a sound
slot is guaranteed to stay valid regardless of playback status. This also means that you must be sure to destroy it when it is no
longer in use, as that slot will otherwise be locked and not available for reuse by new sounds.

Example:
#include  "sound_pool.bgt"

sound_pool  sound_environment;
timer  walk_timer;

int  player_position;
int  step;

const  string  sound_extension=".wav";

const  int  left=-1;
const  int  right=1;

const  int  boundary=200;

void  main()
{
show_game_window("sound_pool  test");
sound_environment.max_distance=70;
for(int  counter=1;  counter<15;  counter++)
{
sound_environment.play_1d("sounds/sound"+counter+sound_extension,  0,  random(0,  boundary),  true);
}
while(true)
{
check_input();

wait(5);
}
}

void  check_input()
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
if(key_down(KEY_LEFT))
{
walk(left);
}
if(key_down(KEY_RIGHT))
{
walk(right);
}
}

void  walk(int  direction)
{
if((direction<=left)&&(player_position<=0))
{
return;
}
if((direction>=right)&&(player_position>=boundary))
{
return;
}
if(walk_timer.elapsed<350)
{
return;
}
step++;
if(step>6)
{
step=1;
}
walk_timer.restart();
player_position+=direction;
sound_environment.play_stationary("sounds/steps/"+step+sound_extension, 
sound_environment.update_listener_1d(player_position);
}

false);

sound_pool object

This method will play a sound that can move around according to the listener and source positions on a 2D grid.

int play_2d(string filename, int listener_x, int listener_y, int sound_x, int sound_y, bool looping, bool persistent=false)

Parameters:
filename
The filename to play.
listener_x
The X position of the listener on the grid.
listener_y
The Y position of the listener on the grid.
sound_x
The X position of the source object on the grid.
sound_y
The Y position of the source object on the grid.
looping
A boolean specifying whether the sound should loop or not.
persistent
An optional boolean specifying whether the sound should be persistent or not, meaning whether or not it should be automatically cleaned up after playback has
finished. If this argument is not given, false is the default which means that the sound does get cleaned up.

Return value:
Returns the slot where the sound is stored in the pool, -1 on error, or -2 if the sound is out of earshot.

Remarks:
This method is useful for playing sounds specific to the game objects and characters, for example monsters, health boosters, teleporters, etc.

A sound that is created with the persistent flag will not be cleaned up after it has finished playing, which otherwise is the default behavior. Instead, an explicit call must
be made to destroy the sound. This is useful for non-looping sounds that you wish to be able to reliably manipulate with any of the functions that use slots to identify a
particular sound. When set as persistent, a sound slot is guaranteed to stay valid regardless of playback status. This also means that you must be sure to destroy it
when it is no longer in use, as that slot will otherwise be locked and not available for reuse by new sounds.

Example:
#include  "sound_pool.bgt"

sound_pool  sound_environment;
timer  walk_timer;

int  player_x,  player_y;
int  step;

const  string  sound_extension=".wav";

const  int  left=-1;
const  int  right=1;

const  int  forward=100;
const  int  back=-100;

const  int  x_boundary=200;
const  int  y_boundary=200;

void  main()
{
show_game_window("sound_pool  test");
sound_environment.max_distance=70;
sound_environment.behind_pitch_decrease=5;

for(int  counter=1;  counter<15;  counter++)
{
sound_environment.play_2d("sounds/sound"+counter+sound_extension,  0,  0,  random(0,  x_boundary),  random(0,  y_boundary),  true);
}
while(true)
{
check_input();
wait(5);
}
}

void  check_input()
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
if(key_down(KEY_LEFT))
{
walk(left);
}
if(key_down(KEY_RIGHT))
{
walk(right);
}
if(key_down(KEY_UP))
{
walk(forward);
}
if(key_down(KEY_DOWN))
{
walk(back);
}
}

void  walk(int  direction)
{
bool  change_y=false;
if(direction>=forward)
{
direction=1;
change_y=true;
}
if(direction<=back)
{
direction=-1;
change_y=true;
}
if(direction<=left)
{
if(change_y==false)
{
if(player_x<=0)
{
return;
}
}
else
{
if(player_y<=0)
{
return;
}
}
}
if(direction>=right)
{
if(change_y==false)
{
if(player_x>=x_boundary)

{
return;
}
}
else
{
if(player_y>=y_boundary)
{
return;
}
}
}
if(walk_timer.elapsed<350)
{
return;
}
step++;
if(step>6)
{
step=1;
}
walk_timer.restart();
if(change_y==false)
{
player_x+=direction;
}
else
{
player_y+=direction;
}
sound_environment.play_stationary("sounds/steps/"+step+sound_extension, 
sound_environment.update_listener_2d(player_x,  player_y);
}

false);

sound_pool object

This method will play a sound that can move around according to the listener and source positions on a 1d grid, allowing you to specify various starting settings for the sound, as
well as a range in which the sound should remain with the listener.

int play_extended_1d(string filename, int listener_x, int sound_x, int left_range, int right_range, bool looping, double offset, float start_pan, float start_volume, float start_pitch, bool
persistent=false)

Parameters:
filename
The filename to play.
listener_x
The position of the listener on the grid.
sound_x
The position of the source object on the grid.
left_range
The leftmost position from the centre of the source where the sound should remain with the listener.
right_range
The rightmost position from the centre of the source where the sound should remain with the listener.
looping
A boolean specifying whether the sound should loop or not.
offset
The point in the sound file where playback should begin, in milliseconds.
start_pan
The starting pan of the sound.
start_volume
The starting volume of the sound.
start_pitch
The starting pitch of the sound.
persistent
An optional boolean specifying whether the sound should be persistent or not, meaning whether or not it should be automatically cleaned up after playback has finished. If this
argument is not given, false is the default which means that the sound does get cleaned up.

Return value:
Returns the slot where the sound is stored in the pool, -1 on error, or -2 if the sound is out of earshot.

Remarks:
This method is useful for playing sounds specific to the game objects and characters, for example monsters, health boosters, teleporters, etc.

The range of the sound determines how far the listener can move before they move away from the source. This, for example, enables you to have a river or an impassable wall of
fire or electricity, etc.

All the position calculations will be done relative to the values you specify for start_pan and start_volume. The pitch will simply be set to the value specified and will then remain
unchanged.

A sound that is created with the persistent flag will not be cleaned up after it has finished playing, which otherwise is the default behavior. Instead, an explicit call must be made to
destroy the sound. This is useful for non-looping sounds that you wish to be able to reliably manipulate with any of the functions that use slots to identify a particular sound. When
set as persistent, a sound slot is guaranteed to stay valid regardless of playback status. This also means that you must be sure to destroy it when it is no longer in use, as that slot
will otherwise be locked and not available for reuse by new sounds.

Example:
#include  "sound_pool.bgt"

sound_pool  sound_environment;
timer  walk_timer;

int  player_position;
int  step;

const  string  sound_extension=".wav";

const  int  left=-1;
const  int  right=1;

const  int  boundary=200;

void  main()
{
show_game_window("sound_pool  test");
sound_environment.max_distance=70;
for(int  counter=1;  counter<15;  counter++)
{
sound_environment.play_extended_1d("sounds/sound"+counter+sound_extension,  0,  random(0,  boundary),  5,  5,  true,  500,  0,  -1,  90,  false);
}
while(true)
{
check_input();
wait(5);
}
}

void  check_input()
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
if(key_down(KEY_LEFT))
{
walk(left);
}
if(key_down(KEY_RIGHT))
{
walk(right);
}
}

void  walk(int  direction)
{
if((direction<=left)&&(player_position<=0))
{
return;
}
if((direction>=right)&&(player_position>=boundary))
{
return;
}
if(walk_timer.elapsed<350)
{
return;
}
step++;
if(step>6)
{
step=1;
}
walk_timer.restart();
player_position+=direction;
sound_environment.play_stationary("sounds/steps/"+step+sound_extension, 
sound_environment.update_listener_1d(player_position);
}

false);

sound_pool object

This method will play a sound that can move around according to the listener and source positions on a 2D grid, allowing you to specify various default settings for the sound, as well as a range in which the sound should
remain with the listener.

int play_extended_2d(string filename, int listener_x, int listener_y, int sound_x, int sound_y, int left_range, int right_range, int backward_range, int forward_range, bool looping, double offset, float start_pan, float start_volume,
float start_pitch, bool persistent=false)

Parameters:
filename
The filename to play.
listener_x
The X position of the listener on the grid.
listener_y
The Y position of the listener on the grid.
sound_x
The X position of the source object on the grid.
sound_y
The Y position of the source object on the grid.
left_range
The leftmost position from the centre of the source where the sound should remain with the listener.
right_range
The rightmost position from the centre of the source where the sound should remain with the listener.
backward_range
The backward position from the centre of the source where the sound should remain with the listener.
forward_range
The forward position from the centre of the source where the sound should remain with the listener.
looping
A boolean specifying whether the sound should loop or not.
offset
The point in the sound file where playback should begin, in milliseconds.
start_pan
The starting pan of the sound.
start_volume
The starting volume of the sound.
start_pitch
The starting pitch of the sound.
persistent
An optional boolean specifying whether the sound should be persistent or not, meaning whether or not it should be automatically cleaned up after playback has finished. If this argument is not given, false is the default which
means that the sound does get cleaned up.

Return value:
Returns the slot where the sound is stored in the pool, -1 on error, or -2 if the sound is out of earshot.

Remarks:
This method is useful for playing sounds specific to the game objects and characters, for example monsters, health boosters, teleporters, etc.

The range of the sound determines how far the listener can move before they move away from the source. This, for example, enables you to have a river or an impassable wall of fire or electricity, etc.

All the position calculations will be done relative to the values you specify for start_pan, start_volume and start_pitch.

A sound that is created with the persistent flag will not be cleaned up after it has finished playing, which otherwise is the default behavior. Instead, an explicit call must be made to destroy the sound. This is useful for non-
looping sounds that you wish to be able to reliably manipulate with any of the functions that use slots to identify a particular sound. When set as persistent, a sound slot is guaranteed to stay valid regardless of playback status.
This also means that you must be sure to destroy it when it is no longer in use, as that slot will otherwise be locked and not available for reuse by new sounds.

Example:
#include  "sound_pool.bgt"

sound_pool  sound_environment;
timer  walk_timer;

int  player_x,  player_y;
int  step;

const  string  sound_extension=".wav";

const  int  left=-1;
const  int  right=1;

const  int  forward=100;
const  int  back=-100;

const  int  x_boundary=200;
const  int  y_boundary=200;

void  main()
{
show_game_window("sound_pool  test");
sound_environment.max_distance=70;
sound_environment.behind_pitch_decrease=5;
for(int  counter=1;  counter<15;  counter++)
{
sound_environment.play_extended_2d("sounds/sound"+counter+sound_extension,  0,  0,  random(0,  x_boundary),  random(0,  y_boundary),  5,  5,  5,  5,  true,  500,  0,  -1,  90,  false);
}
while(true)
{
check_input();
wait(5);
}
}

void  check_input()
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
if(key_down(KEY_LEFT))
{
walk(left);
}
if(key_down(KEY_RIGHT))
{
walk(right);
}
if(key_down(KEY_UP))
{
walk(forward);
}
if(key_down(KEY_DOWN))
{
walk(back);
}
}

void  walk(int  direction)
{
bool  change_y=false;
if(direction>=forward)
{
direction=1;
change_y=true;
}

if(direction<=back)
{
direction=-1;
change_y=true;
}
if(direction<=left)
{
if(change_y==false)
{
if(player_x<=0)
{
return;
}
}
else
{
if(player_y<=0)
{
return;
}
}
}
if(direction>=right)
{
if(change_y==false)
{
if(player_x>=x_boundary)
{
return;
}
}
else
{
if(player_y>=y_boundary)
{
return;
}
}
}
if(walk_timer.elapsed<350)
{
return;
}
step++;
if(step>6)
{
step=1;
}
walk_timer.restart();
if(change_y==false)
{
player_x+=direction;
}
else
{
player_y+=direction;
}
sound_environment.play_stationary("sounds/steps/"+step+sound_extension, 
sound_environment.update_listener_2d(player_x,  player_y);
}

false);

sound_pool object

This method will play a sound that will remain central at the listener's position at all times.

int play_stationary(string filename, bool looping, bool persistent=false)

Parameters:
filename
The filename to play.
looping
A boolean specifying whether the sound should loop or not.
persistent
An optional boolean specifying whether the sound should be persistent or not, meaning whether or not it should be automatically
cleaned up after playback has finished. If this argument is not given, false is the default which means that the sound does get
cleaned up.

Return value:
Returns the slot where the sound is stored in the pool or -1 on error.

Remarks:
This method is useful for playing sounds such as the listener's footsteps, game ambiences, etc.

A sound that is created with the persistent flag will not be cleaned up after it has finished playing, which otherwise is the default
behavior. Instead, an explicit call must be made to destroy the sound. This is useful for non-looping sounds that you wish to be
able to reliably manipulate with any of the functions that use slots to identify a particular sound. When set as persistent, a sound
slot is guaranteed to stay valid regardless of playback status. This also means that you must be sure to destroy it when it is no
longer in use, as that slot will otherwise be locked and not available for reuse by new sounds.

Example:
#include  "sound_pool.bgt"

sound_pool  sound_environment;
timer  walk_timer;

int  player_position;
int  step;

const  string  sound_extension=".wav";

const  int  left=-1;
const  int  right=1;

const  int  boundary=200;

void  main()
{
show_game_window("sound_pool  test");
sound_environment.max_distance=70;
for(int  counter=1;  counter<15;  counter++)
{
sound_environment.play_1d("sounds/sound"+counter+sound_extension,  0,  random(0,  boundary),  true);
}
while(true)
{
check_input();
wait(5);
}
}

void  check_input()
{

if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
if(key_down(KEY_LEFT))
{
walk(left);
}
if(key_down(KEY_RIGHT))
{
walk(right);
}
}

void  walk(int  direction)
{
if((direction<=left)&&(player_position<=0))
{
return;
}
if((direction>=right)&&(player_position>=boundary))
{
return;
}
if(walk_timer.elapsed<350)
{
return;
}
step++;
if(step>6)
{
step=1;
}
walk_timer.restart();
player_position+=direction;
sound_environment.play_stationary("sounds/steps/"+step+sound_extension, 
sound_environment.update_listener_1d(player_position);
}

false);

sound_pool object

This method will play a sound that will remain central at the listener's position at all times, and allows you to choose starting values for the
playback of the file.

int play_stationary_extended(string filename, bool looping, double offset, float start_pan, float start_volume, float start_pitch, bool
persistent=false)

Parameters:
filename
The filename to play.
looping
A boolean specifying whether the sound should loop or not.
offset
The point in the sound file where playback should begin, in milliseconds.
start_pan
The starting pan of the sound.
start_volume
The starting volume of the sound.
start_pitch
The starting pitch of the sound.
persistent
An optional boolean specifying whether the sound should be persistent or not, meaning whether or not it should be automatically cleaned up
after playback has finished. If this argument is not given, false is the default which means that the sound does get cleaned up.

Return value:
Returns the slot where the sound is stored in the pool or -1 on error.

Remarks:
This method is useful for playing sounds such as the listener's footsteps, game ambiences, etc.

A sound that is created with the persistent flag will not be cleaned up after it has finished playing, which otherwise is the default behavior.
Instead, an explicit call must be made to destroy the sound. This is useful for non-looping sounds that you wish to be able to reliably manipulate
with any of the functions that use slots to identify a particular sound. When set as persistent, a sound slot is guaranteed to stay valid regardless
of playback status. This also means that you must be sure to destroy it when it is no longer in use, as that slot will otherwise be locked and not
available for reuse by new sounds.

Example:
#include  "sound_pool.bgt"

sound_pool  sound_environment;
timer  walk_timer;

int  player_position;
int  step;

const  string  sound_extension=".wav";

const  int  left=-1;
const  int  right=1;

const  int  boundary=200;

void  main()
{
show_game_window("sound_pool  test");
sound_environment.max_distance=70;
sound_environment.play_stationary_extended("sounds/ambience"+sound_extension,  true,  500,  0,  -1,  90,  false);
for(int  counter=1;  counter<15;  counter++)

{
sound_environment.play_1d("sounds/sound"+counter+sound_extension,  0,  random(0,  boundary),  true);
}
while(true)
{
check_input();
wait(5);
}
}

void  check_input()
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
if(key_down(KEY_LEFT))
{
walk(left);
}
if(key_down(KEY_RIGHT))
{
walk(right);
}
}

void  walk(int  direction)
{
if((direction<=left)&&(player_position<=0))
{
return;
}
if((direction>=right)&&(player_position>=boundary))
{
return;
}
if(walk_timer.elapsed<350)
{
return;
}
step++;
if(step>6)
{
step=1;
}
walk_timer.restart();
player_position+=direction;
sound_environment.play_stationary("sounds/steps/"+step+sound_extension, 
sound_environment.update_listener_1d(player_position);
}

false);

sound_pool object

This method restarts all of the paused sounds in the sound pool.

void resume_all()

Parameters:
None.

Return value:
None.

Remarks:
None.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;

void  main()
{
sounds.max_distance=70;
sounds.play_stationary("sounds/ambience.wav",  true);
wait(5000);
sounds.resume_all();
wait(5000);
sounds.resume_all();
wait(5000);
}

sound_pool object

This method restarts a sound that has been previously paused.

bool resume_sound(int slot)

Parameters:
slot
The slot of the sound to resume.

Return value:
true on success, false on failure.

Remarks:
Please note: When dealing with sound slots, be sure that you set the persistent flag to true for all non-looping sounds when you
first create them. If you fail to do this, manipulating a sound in any way by use of its slot number can have unpredictable results.
This is because the sound pool automatically cleans up any sound that has finished playing and that is not set to be persistent,
with the result that the slot that was returned on creation is no longer invalid and may, in the worst case scenario, refer to a
completely different sound.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;

void  main()
{
sounds.max_distance=70;
int  slot=sounds.play_stationary("sounds/ambience.wav",  true);
wait(5000);
sounds.resume_sound(slot);
wait(5000);
sounds.resume_sound(slot);
wait(5000);
}

sound_pool object

This method checks to see whether a given slot contains a valid sound.

bool sound_is_active(int slot)

Parameters:
slot
The slot of a sound, as returned by one of the play methods.

Return value:
True if the slot contains an active sound, false otherwise.

Remarks:
This is useful if you want to check which sounds you are able to successfully modify before changing any of their properties or
positions.

Please note: When dealing with sound slots, be sure that you set the persistent flag to true for all non-looping sounds when you
first create them. If you fail to do this, manipulating a sound in any way by use of its slot number can have unpredictable results.
This is because the sound pool automatically cleans up any sound that has finished playing and that is not set to be persistent,
with the result that the slot that was returned on creation is no longer invalid and may, in the worst case scenario, refer to a
completely different sound.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;

void  main()
{
sounds.max_distance=70;
int  slot=sounds.play_1d("sounds/test.wav",  0,  20,  true);
if(sounds.sound_is_active(slot))
{
alert("information",  "slot  "+slot+"  is  currently  active.");
}
else
{
alert("information",  "slot  "+slot+"  is  not  currently  active.");
}
}

sound_pool object

This method checks to see whether a sound is playing in a given slot.

bool sound_is_playing(int slot)

Parameters:
slot
The slot of a sound, as returned by one of the play methods.

Return value:
True if the sound assigned to the given slot is playing, false otherwise.

Remarks:
Please note that the sound will report as not playing if it is either closed or out of earshot. To check whether a sound exists, use
the sound_is_active method.

This is useful if you want to perform another action if a sound is not playing.

Please note: When dealing with sound slots, be sure that you set the persistent flag to true for all non-looping sounds when you
first create them. If you fail to do this, manipulating a sound in any way by use of its slot number can have unpredictable results.
This is because the sound pool automatically cleans up any sound that has finished playing and that is not set to be persistent,
with the result that the slot that was returned on creation is no longer invalid and may, in the worst case scenario, refer to a
completely different sound.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;

void  main()
{
sounds.max_distance=70;
int  slot=sounds.play_1d("sounds/test.wav",  0,  20,  true);
if(sounds.sound_is_playing(slot))
{
alert("information",  "slot  "+slot+"  is  currently  playing.");
}
else
{
alert("information",  "slot  "+slot+"  is  not  currently  playing.");
}
}

sound_pool object

This method will update the position of the sounds according to the listener's new position on a 1d grid.

void update_listener_1d(int listener_x)

Parameters:
listener_x
The new position of the listener on the grid.

Return value:
None.

Remarks:
This method is used when the listener has moved, meaning that all of the currently active sounds in the pool will move. To move individual sounds, use the
update_sound methods.

Example:
#include  "sound_pool.bgt"

sound_pool  sound_environment;
timer  walk_timer;

int  player_position;
int  step;

const  string  sound_extension=".wav";

const  int  left=-1;
const  int  right=1;

const  int  boundary=200;

void  main()
{
show_game_window("sound_pool  test");
sound_environment.max_distance=70;
for(int  counter=1;  counter<15;  counter++)
{
sound_environment.play_extended_1d("sounds/sound"+counter+sound_extension,  0,  random(0,  boundary),  -10,  10,  true,  0,  false);
}
while(true)
{
check_input();
wait(5);
}
}

void  check_input()
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
if(key_down(KEY_LEFT))
{
walk(left);
}
if(key_down(KEY_RIGHT))
{
walk(right);
}
}

void  walk(int  direction)
{
if((direction<=left)&&(player_position<=0))
{
return;
}
if((direction>=right)&&(player_position>=boundary))
{
return;
}
if(walk_timer.elapsed<350)
{
return;
}
step++;
if(step>6)
{
step=1;
}
walk_timer.restart();
player_position+=direction;
sound_environment.play_stationary("sounds/steps/"+step+sound_extension, 
sound_environment.update_listener_1d(player_position);
}

false);

sound_pool object

This method will update the position of the sounds according to the listener's new position on a 2d grid.

void update_listener_2d(int listener_x, int listener_y)

Parameters:
listener_x
The new X position of the listener on the grid.
listener_y
The new Y position of the listener on the grid.

Return value:
None.

Remarks:
This method is used when the listener has moved, meaning that all of the currently active sounds in the pool will move. To move individual sounds, use the
update_sound methods.

Example:
#include  "sound_pool.bgt"

sound_pool  sound_environment;
timer  walk_timer;

int  player_x,  player_y;
int  step;

const  string  sound_extension=".wav";

const  int  left=-1;
const  int  right=1;

const  int  forward=100;
const  int  back=-100;

const  int  x_boundary=200;
const  int  y_boundary=200;

void  main()
{
show_game_window("sound_pool  test");
sound_environment.max_distance=70;
sound_environment.behind_pitch=95;
for(int  counter=1;  counter<15;  counter++)
{
sound_environment.play_2d("sounds/sound"+counter+sound_extension,  0,  0,  random(0,  x_boundary),  random(0,  y_boundary),  true);
}
while(true)
{
check_input();
wait(5);
}
}

void  check_input()
{
if((key_down(KEY_LMENU))&&(key_pressed(KEY_F4)))
{
exit();
}
if(key_down(KEY_LEFT))
{

walk(left);
}
if(key_down(KEY_RIGHT))
{
walk(right);
}
if(key_down(KEY_UP))
{
walk(forward);
}
if(key_down(KEY_DOWN))
{
walk(back);
}
}

void  walk(int  direction)
{
bool  change_y=false;
if(direction>=forward)
{
direction=1;
change_y=true;
}
if(direction<=back)
{
direction=-1;
change_y=true;
}
if(direction<=left)
{
if(change_y==false)
{
if(player_x<=0)
{
return;
}
}
else
{
if(player_y<=0)
{
return;
}
}
}
if(direction>=right)
{
if(change_y==false)
{
if(player_x>=x_boundary)
{
return;
}
}
else
{
if(player_y>=y_boundary)
{
return;
}
}
}
if(walk_timer.elapsed<350)
{
return;
}
step++;
if(step>6)
{

step=1;
}
walk_timer.restart();
if(change_y==false)
{
player_x+=direction;
}
else
{
player_y+=direction;
}
sound_environment.play_stationary("sounds/steps/"+step+sound_extension, 
sound_environment.update_listener_2d(player_x,  player_y);
}

false);

sound_pool object

This method will update the position of a sound on a 1d grid.

bool update_sound_1d(int slot, int x)

Parameters:
slot
The slot of the sound to update.
x
The new position of the sound.

Return value:
true on success, false on failure.

Remarks:
This method will update the position of individual sounds. To update all the sounds based on the listener's position, use the
update_listener methods instead.

Please note: When dealing with sound slots, be sure that you set the persistent flag to true for all non-looping sounds when you
first create them. If you fail to do this, manipulating a sound in any way by use of its slot number can have unpredictable results.
This is because the sound pool automatically cleans up any sound that has finished playing and that is not set to be persistent,
with the result that the slot that was returned on creation is no longer invalid and may, in the worst case scenario, refer to a
completely different sound.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;
timer  move_timer;

int  pos;

void  main()
{
sounds.max_distance=70;
pos=0;
int  slot=sounds.play_1d("sounds/loop.wav",  50,  pos,  true);
while(true)
{
if(move_timer.elapsed>=200)
{
move_timer.restart();
pos++;
sounds.update_sound_1d(slot,  pos);
}
if(pos>=100)
{
exit();
}
}
}

sound_pool object

This method will update the position of a sound on a 2d grid.

bool update_sound_2d(int slot, int x, int y)

Parameters:
slot
The slot of the sound to update.
x
The new X position of the sound.
y
The new Y position of the sound.

Return value:
true on success, false on failure.

Remarks:
This method will update the position of individual sounds. To update all the sounds based on the listener's position, use the
update_listener methods instead.

Please note: When dealing with sound slots, be sure that you set the persistent flag to true for all non-looping sounds when you
first create them. If you fail to do this, manipulating a sound in any way by use of its slot number can have unpredictable results.
This is because the sound pool automatically cleans up any sound that has finished playing and that is not set to be persistent,
with the result that the slot that was returned on creation is no longer invalid and may, in the worst case scenario, refer to a
completely different sound.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;
timer  move_timer;

int x, y;

void  main()
{
sounds.max_distance=70;
sounds.behind_pitch_decrease=5;
x=0;
y=40;
int  slot=sounds.play_2d("sounds/loop.wav",  20,  20,  x,  y,  true);
while(true)
{
if(move_timer.elapsed>=200)
{
move_timer.restart();
x++;
y--;
sounds.update_sound_2d(slot,  x,  y);
}
if(x>=40)
{
exit();
}
}
}

sound_pool object

This method will update the range of a sound on a 1d grid.

bool update_sound_range_1d(int slot, int left_range, int right_range)

Parameters:
slot
The slot of the sound to update.
left_range
The leftmost position from the centre of the source where the sound should remain with the listener.
right_range
The rightmost position from the centre of the source where the sound should remain with the listener.

Return value:
true on success, false on failure.

Remarks:
The range of the sound determines how far the listener can move before they move away from the source. This, for example,
enables you to have a river or an impassable wall of fire or electricity, etc.

Please note: When dealing with sound slots, be sure that you set the persistent flag to true for all non-looping sounds when you
first create them. If you fail to do this, manipulating a sound in any way by use of its slot number can have unpredictable results.
This is because the sound pool automatically cleans up any sound that has finished playing and that is not set to be persistent,
with the result that the slot that was returned on creation is no longer invalid and may, in the worst case scenario, refer to a
completely different sound.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;
timer  move_timer;

int  pos;

void  main()
{
sounds.max_distance=70;
pos=0;
int  slot=sounds.play_1d("sounds/loop.wav",  100,  pos,  true);
while(true)
{
if(move_timer.elapsed>=200)
{
move_timer.restart();
pos++;
sounds.update_sound_range_1d(slot,  5,  5);
sounds.update_sound_1d(slot,  pos);
}
if(pos>=100)
{
exit();
}
}
}

sound_pool object

This method will update the range of a sound on a 2d grid.

bool update_sound_range_2d(int slot, int left_range, int right_range, int backward_range, int forward_range)

Parameters:
slot
The slot of the sound to update.
left_range
The leftmost position from the centre of the source where the sound should remain with the listener.
right_range
The rightmost position from the centre of the source where the sound should remain with the listener.
backward_range
The backward position from the centre of the source where the sound should remain with the listener.
forward_range
The forward position from the centre of the source where the sound should remain with the listener.

Return value:
true on success, false on failure.

Remarks:
The range of the sound determines how far the listener can move before they move away from the source. This, for example,
enables you to have a river or an impassable wall of fire or electricity, etc.

Please note: When dealing with sound slots, be sure that you set the persistent flag to true for all non-looping sounds when you
first create them. If you fail to do this, manipulating a sound in any way by use of its slot number can have unpredictable results.
This is because the sound pool automatically cleans up any sound that has finished playing and that is not set to be persistent,
with the result that the slot that was returned on creation is no longer invalid and may, in the worst case scenario, refer to a
completely different sound.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;
timer  move_timer;

int x, y;

void  main()
{
sounds.max_distance=70;
sounds.behind_pitch=95;
x=0;
y=0;
int  slot=sounds.play_1d("sounds/loop.wav",  100,  pos,  true);
while(true)
{
if(move_timer.elapsed>=200)
{
move_timer.restart();
x++;
y++;
sounds.update_sound_range_2d(slot,  5,  5,  5,  5);
sounds.update_sound_2d(slot,  x,  y);
}
if(x>=100)
{
exit();
}

}
}

sound_pool object

This method will update the start values of the sound.

bool update_sound_start_values(int slot, float start_pan, float start_volume, float start_pitch)

Parameters:
slot
The slot of the sound to update.
start_pan
The new starting pan of the sound.
start_volume
The new starting volume of the sound.
start_pitch
The new starting pitch of the sound.

Return value:
true on success, false on failure.

Remarks:
All the position calculations will be done relative to the values you specify for start_pan and start_volume, and for 2d sounds this
also applies to start_pitch. For 1d sounds, the pitch will simply be set to the value specified and will then remain unchanged.

The parameters of this function expect absolute values.

Please note: When dealing with sound slots, be sure that you set the persistent flag to true for all non-looping sounds when you
first create them. If you fail to do this, manipulating a sound in any way by use of its slot number can have unpredictable results.
This is because the sound pool automatically cleans up any sound that has finished playing and that is not set to be persistent,
with the result that the slot that was returned on creation is no longer invalid and may, in the worst case scenario, refer to a
completely different sound.

Example:
#include  "sound_pool.bgt"

sound_pool  sounds;
timer  move_timer;

int  pos;

void  main()
{
sounds.max_distance=70;
pos=0;
int  slot=sounds.play_1d("sounds/loop.wav",  100,  pos,  true);
while(true)
{
if(move_timer.elapsed>=200)
{
move_timer.restart();
pos++;
sounds.update_sound_1d(slot,  pos);
sounds.update_sound_start_values(slot,  0,  -1,  90);
}
if(pos>=100)
{
exit();
}
}
}

sound_pool object

int max_distance
The maximum distance at which to play a sound. It is essential to set this property if a larger quantity of sounds are to play. This
will close any sounds that are out of earshot (e.g. beyond the value of max_distance), to improve performance. A value of 0
which is the default, turns off this optimization.

float pan_step
The value that specifies how much the sound should pan when the object and/or listener position changes. Changing this value
will only affect sound sources created in subsequent calls to one of the play methods, but not existing sources.

float volume_step
The value that specifies how much the sound's volume will change when the object and/or listener position changes. Changing this
value will only affect sound sources created in subsequent calls to one of the play methods, but not existing sources.

float behind_pitch_decrease
The pitch that will simulate the object sound behind the listener relative to the current pitch. Changing this value will only affect
sound sources created in subsequent calls to one of the play methods, but not existing sources.

soundtrack object

The soundtrack object is stored in soundtrack.bgt in the BGT include directory. Building upon the BGT tone generator, the soundtrack object attempts to give you more music for less code.

soundtrack()

Parameters:
None.

Remarks:
Using the soundtrack object, you create music by adding channels to a song. A channel is described by a sequence of commands. Note that the command sequence contains no spaces---it
is simply a string of commands as defined below.

Setting the octave

Use the o command followed by a number to set the octave for subsequent notes. Valid octaves are 1 to 6, so o1 through o6 are valid octave commands. The default octave starts out as 4.

Setting the default length

Use the l command followed by a number to set the default length for subsequent notes. A note may provide its own length, but those that don't will get the default length. Lengths are
specified by how many of them would fit into a whole note. For example, 1 is a whole note, 2 is a half note, and 16 is a 16th note. Any whole number from 1 to 64 is allowed. For example,
24 is a length that would fit three times into an 8th note. The default length starts out as 4 (meaning quarter notes).

Increasing or decreasing the octave

The < and > commands provide convenient shortcuts for increasing or decreasing the octave by one. The > command increases the octave if possible, and the < command decreases it if
possible.

Selecting the instrument

Use the @ command followed by a number to select the instrument for subsequent notes. Valid instruments are @1 (sawtooth), @2 (square), @3 (sine) and @4 (triangle). The instrument
starts out as @1 (sawtooth).

Setting the quantization

Quantization determines how much of the note will actually be heard. Use the q command followed by a number between 1 and 100 to set the quantization for subsequent notes. A
quantization of 8 means the entire note will be heard while, for example, a quantization of 4 means that only the first half of the note will be heard whereas the second half will be silenced.
Use quantizations greater than 8 to make a note last beyond its notational end. The quantization starts out at 8.

Setting the tempo

Use the t command followed by a number to set the tempo for subsequent notes. The tempo is measured in quarter notes per minute, and valid tempi are numbers between 60 and 240. The
tempo starts out as 120. You may have as many tempo changes as you like, and it is perfectly valid to have different channels play at different tempi although the result might not be to your
liking.

Setting the attack

Attack determines how long the note will rise in volume until it reaches its maximum. Use the a command followed by a number to set the attack for subsequent notes. Valid attack values are
0 to 100. 100 means that the note will take a second to reach its maximum volume, while 0 means that the note will start out at its maximum volume. The attack value starts out as 1.

Setting the release

Release determines how long the note will take to fade from its maximum volume to silence. Use the r command to set the release for subsequent notes. Valid release values are 0 to 100.
100 means that the note will take a second to fade, while 0 means it will go directly to silence as if cut off. Note that having a note with 0 release followed by a note with 0 attack may create
a clicking noise. This is not a bug but simply means the generator is doing what it should. The release value starts out as 1.

Generating a pause

Use the p command followed by a number to insert a pause, or rest, into your channel. The number follows the same rules as that used in the l command, so p1 will create a whole rest, p4
will create a quarter rest, etc.

Setting the volume

Use v followed by a number between 0 and 100 to set the volume. 100 is loudest while 0 is silent. The volume starts out at 100.

Transposition

Use u followed by a number to shift subsequent notes up the given number of semitones. Use d followed by a number to shift subsequent notes down the given number of semitones. Use the
n command to specify that subsequent notes should no longer be transposed. The u and d commands accept numbers from 1 to 11. These three commands are best remembered as u for up,
d for down, and n for normal.

Generating notes

Use one of the uppercase letters C, D, E, F, G, A, B to insert a corresponding note. To sharpen the note, put a + after it, and to flatten the note, put a - after it.
If you leave it at that, the note will get the default length. To change the length for just this single note, put a number directly after it. For example, E-2 is a half E flat.
Finally, you can put a . after the whole thing to get a dotted note. A dotted note is simply a note with its length increased by half of what it was. For example, F2. is a half F dotted, meaning
an F the length of three quarter notes.

Generating chords

To generate chords, i.e. simultaneous notes, connect them with an & sign, like so:

C&E&G
This would generate a C major chord in the current channel.
It is valid to connect notes in this way even if they are of different lengths. Technically, the & operator only makes sure that all the notes it connects start at the same time. Consider the
following:
l1C&E&G&>l8CDEFGAG4
In terms of a piano, this is a left hand playing a C major chord while the right hand, one octave higher, plays a little cheery melody.

Repetitions

To repeat a sequence of commands a given number of times, enclose the sequence in brackets and put a number from 2 to 100 directly after the closing bracket, like so:
[CDE]3
This would be equivalent to:
CDECDECDE
Any sequence of commands is valid between the brackets. For example, to shift down 3 octaves you can write:
[<]3
Repetitions can even be nested. For example:
[[CDE]3[EFG]3]2
is equivalent to:
CDECDECDEEFGEFGEFGCDECDECDEEFGEFGEFG

Example:
// Write a song.

#include  "soundtrack.bgt"

void  main()
{
soundtrack  s;
string  preamble  =  "t140u6";
string  chords1  =  "@1v97l2a8r100q12[C&E&G]2[<B&>E&G]2[<B-&>E&G]2[<A&>D&F]2<F+&A&B&>D<A-&B&>D&E<A&>C&E<G&>C&E<E&G&>C<D&F+&>C<F&G&>C&D<F&G&B&>D";
string  chords2  =  "[<B&>D&G]2[C&E&G]2q20<B4&>D4&B4C8&E8&>C8<<l4B&>D&B<A&>C&Aq12l2<G&B&>Gp8<G&B&>C&G<B&>D&G<B&>D&F<B&>D&G<G&>C&E";
string  chords3  =  "<G&>C&E<F+&>C&D<F&>C&D<F&B&>D";
string  chords4  =  "[<A&>C&F]3<G&B&>F<G&>C&El8q8r10<B-&>C&Gl4<A-&>C&Fl1<G&>C&Ep8l4<A-&>C&E<A-&>C&Dp4l1<E&G&>C";
string chords = chords1 + chords1 + chords2 + chords3 + chords2 + chords4;
string  bass1  =  "@3r100C2.CC2.CC2.CC2.C<B2>E2<A2.A>D2.D<G2.";
string  bass2  =  "E2.E<A2.A>F2.FE2<A2>F2<B2>E2<A2>";
string  bass3  =  "D2.D<G2.>F";
string  bass4  =  "D2.D4<G2.G>C1C1C4<G4p4C1";
string  bass  =  bass1  +  "G>"  +  bass1  +  ">F"  +  bass2  +  bass3  +  bass2  +  bass4;
string  melody1  =  "@2a10r10E2.p4E2.p4EE8FCD2.p4p8DD8E4.<B8>C4.DC<BA2p8>E.";
string  melody2  =  "l4G2.p4G2.p4B>C8<BAG2.p4p8B8B8>C8<B4.A8G4.>D8C4.<";
string  melody3  =  "E8E2.p8D8CD2A4.";
string  melody4  =  "l4EEF8F4.GA2B2>C1.";
string melody = melody1 + "D2.p4" + melody1 + "D2p8A4." + melody2 + melody3 + melody2 + melody4;
s.add_channel(preamble  +  "o3"  +  bass);
s.add_channel(preamble  +  "o2"  +  bass);
s.add_channel(preamble  +  chords);
s.add_channel(preamble  +  melody);
s.add_channel(preamble  +  "o5v90"  +  melody);
s.write("song4.wav");
}

soundtrack object

The soundtrack object is stored in soundtrack.bgt in the BGT include directory. Building upon the BGT tone generator, the soundtrack object attempts to give you more music for less code.

soundtrack()

Parameters:
None.

Remarks:
Using the soundtrack object, you create music by adding channels to a song. A channel is described by a sequence of commands. Note that the command sequence contains no spaces---it
is simply a string of commands as defined below.

Setting the octave

Use the o command followed by a number to set the octave for subsequent notes. Valid octaves are 1 to 6, so o1 through o6 are valid octave commands. The default octave starts out as 4.

Setting the default length

Use the l command followed by a number to set the default length for subsequent notes. A note may provide its own length, but those that don't will get the default length. Lengths are
specified by how many of them would fit into a whole note. For example, 1 is a whole note, 2 is a half note, and 16 is a 16th note. Any whole number from 1 to 64 is allowed. For example,
24 is a length that would fit three times into an 8th note. The default length starts out as 4 (meaning quarter notes).

Increasing or decreasing the octave

The < and > commands provide convenient shortcuts for increasing or decreasing the octave by one. The > command increases the octave if possible, and the < command decreases it if
possible.

Selecting the instrument

Use the @ command followed by a number to select the instrument for subsequent notes. Valid instruments are @1 (sawtooth), @2 (square), @3 (sine) and @4 (triangle). The instrument
starts out as @1 (sawtooth).

Setting the quantization

Quantization determines how much of the note will actually be heard. Use the q command followed by a number between 1 and 100 to set the quantization for subsequent notes. A
quantization of 8 means the entire note will be heard while, for example, a quantization of 4 means that only the first half of the note will be heard whereas the second half will be silenced.
Use quantizations greater than 8 to make a note last beyond its notational end. The quantization starts out at 8.

Setting the tempo

Use the t command followed by a number to set the tempo for subsequent notes. The tempo is measured in quarter notes per minute, and valid tempi are numbers between 60 and 240. The
tempo starts out as 120. You may have as many tempo changes as you like, and it is perfectly valid to have different channels play at different tempi although the result might not be to your
liking.

Setting the attack

Attack determines how long the note will rise in volume until it reaches its maximum. Use the a command followed by a number to set the attack for subsequent notes. Valid attack values are
0 to 100. 100 means that the note will take a second to reach its maximum volume, while 0 means that the note will start out at its maximum volume. The attack value starts out as 1.

Setting the release

Release determines how long the note will take to fade from its maximum volume to silence. Use the r command to set the release for subsequent notes. Valid release values are 0 to 100.
100 means that the note will take a second to fade, while 0 means it will go directly to silence as if cut off. Note that having a note with 0 release followed by a note with 0 attack may create
a clicking noise. This is not a bug but simply means the generator is doing what it should. The release value starts out as 1.

Generating a pause

Use the p command followed by a number to insert a pause, or rest, into your channel. The number follows the same rules as that used in the l command, so p1 will create a whole rest, p4
will create a quarter rest, etc.

Setting the volume

Use v followed by a number between 0 and 100 to set the volume. 100 is loudest while 0 is silent. The volume starts out at 100.

Transposition

Use u followed by a number to shift subsequent notes up the given number of semitones. Use d followed by a number to shift subsequent notes down the given number of semitones. Use the
n command to specify that subsequent notes should no longer be transposed. The u and d commands accept numbers from 1 to 11. These three commands are best remembered as u for up,
d for down, and n for normal.

Generating notes

Use one of the uppercase letters C, D, E, F, G, A, B to insert a corresponding note. To sharpen the note, put a + after it, and to flatten the note, put a - after it.
If you leave it at that, the note will get the default length. To change the length for just this single note, put a number directly after it. For example, E-2 is a half E flat.
Finally, you can put a . after the whole thing to get a dotted note. A dotted note is simply a note with its length increased by half of what it was. For example, F2. is a half F dotted, meaning
an F the length of three quarter notes.

Generating chords

To generate chords, i.e. simultaneous notes, connect them with an & sign, like so:

C&E&G
This would generate a C major chord in the current channel.
It is valid to connect notes in this way even if they are of different lengths. Technically, the & operator only makes sure that all the notes it connects start at the same time. Consider the
following:
l1C&E&G&>l8CDEFGAG4
In terms of a piano, this is a left hand playing a C major chord while the right hand, one octave higher, plays a little cheery melody.

Repetitions

To repeat a sequence of commands a given number of times, enclose the sequence in brackets and put a number from 2 to 100 directly after the closing bracket, like so:
[CDE]3
This would be equivalent to:
CDECDECDE
Any sequence of commands is valid between the brackets. For example, to shift down 3 octaves you can write:
[<]3
Repetitions can even be nested. For example:
[[CDE]3[EFG]3]2
is equivalent to:
CDECDECDEEFGEFGEFGCDECDECDEEFGEFGEFG

Example:
// Write a song.

#include  "soundtrack.bgt"

void  main()
{
soundtrack  s;
string  preamble  =  "t140u6";
string  chords1  =  "@1v97l2a8r100q12[C&E&G]2[<B&>E&G]2[<B-&>E&G]2[<A&>D&F]2<F+&A&B&>D<A-&B&>D&E<A&>C&E<G&>C&E<E&G&>C<D&F+&>C<F&G&>C&D<F&G&B&>D";
string  chords2  =  "[<B&>D&G]2[C&E&G]2q20<B4&>D4&B4C8&E8&>C8<<l4B&>D&B<A&>C&Aq12l2<G&B&>Gp8<G&B&>C&G<B&>D&G<B&>D&F<B&>D&G<G&>C&E";
string  chords3  =  "<G&>C&E<F+&>C&D<F&>C&D<F&B&>D";
string  chords4  =  "[<A&>C&F]3<G&B&>F<G&>C&El8q8r10<B-&>C&Gl4<A-&>C&Fl1<G&>C&Ep8l4<A-&>C&E<A-&>C&Dp4l1<E&G&>C";
string chords = chords1 + chords1 + chords2 + chords3 + chords2 + chords4;
string  bass1  =  "@3r100C2.CC2.CC2.CC2.C<B2>E2<A2.A>D2.D<G2.";
string  bass2  =  "E2.E<A2.A>F2.FE2<A2>F2<B2>E2<A2>";
string  bass3  =  "D2.D<G2.>F";
string  bass4  =  "D2.D4<G2.G>C1C1C4<G4p4C1";
string  bass  =  bass1  +  "G>"  +  bass1  +  ">F"  +  bass2  +  bass3  +  bass2  +  bass4;
string  melody1  =  "@2a10r10E2.p4E2.p4EE8FCD2.p4p8DD8E4.<B8>C4.DC<BA2p8>E.";
string  melody2  =  "l4G2.p4G2.p4B>C8<BAG2.p4p8B8B8>C8<B4.A8G4.>D8C4.<";
string  melody3  =  "E8E2.p8D8CD2A4.";
string  melody4  =  "l4EEF8F4.GA2B2>C1.";
string melody = melody1 + "D2.p4" + melody1 + "D2p8A4." + melody2 + melody3 + melody2 + melody4;
s.add_channel(preamble  +  "o3"  +  bass);
s.add_channel(preamble  +  "o2"  +  bass);
s.add_channel(preamble  +  chords);
s.add_channel(preamble  +  melody);
s.add_channel(preamble  +  "o5v90"  +  melody);
s.write("song4.wav");
}

soundtrack object

This method adds a channel to a song.

void add_channel(string commands)

Parameters:
commands
a string describing the channel to be added. See the documentation for the soundtrack object for a complete list of all supported commands and their exact syntax.

Return value:
None.

Remarks:
A channel is simply a voice, or instrument, or track, within your song. The song may have as many channels as you like.

Example:
// Write a song.

#include  "soundtrack.bgt"

void  main()
{
soundtrack  s;
string  preamble  =  "t140u6";
string  chords1  =  "@1v97l2a8r100q12[C&E&G]2[<B&>E&G]2[<B-&>E&G]2[<A&>D&F]2<F+&A&B&>D<A-&B&>D&E<A&>C&E<G&>C&E<E&G&>C<D&F+&>C<F&G&>C&D<F&G&B&>D";
string  chords2  =  "[<B&>D&G]2[C&E&G]2q20<B4&>D4&B4C8&E8&>C8<<l4B&>D&B<A&>C&Aq12l2<G&B&>Gp8<G&B&>C&G<B&>D&G<B&>D&F<B&>D&G<G&>C&E";
string  chords3  =  "<G&>C&E<F+&>C&D<F&>C&D<F&B&>D";
string  chords4  =  "[<A&>C&F]3<G&B&>F<G&>C&El8q8r10<B-&>C&Gl4<A-&>C&Fl1<G&>C&Ep8l4<A-&>C&E<A-&>C&Dp4l1<E&G&>C";
string chords = chords1 + chords1 + chords2 + chords3 + chords2 + chords4;
string  bass1  =  "@3r100C2.CC2.CC2.CC2.C<B2>E2<A2.A>D2.D<G2.";
string  bass2  =  "E2.E<A2.A>F2.FE2<A2>F2<B2>E2<A2>";
string  bass3  =  "D2.D<G2.>F";
string  bass4  =  "D2.D4<G2.G>C1C1C4<G4p4C1";
string  bass  =  bass1  +  "G>"  +  bass1  +  ">F"  +  bass2  +  bass3  +  bass2  +  bass4;
string  melody1  =  "@2a10r10E2.p4E2.p4EE8FCD2.p4p8DD8E4.<B8>C4.DC<BA2p8>E.";
string  melody2  =  "l4G2.p4G2.p4B>C8<BAG2.p4p8B8B8>C8<B4.A8G4.>D8C4.<";
string  melody3  =  "E8E2.p8D8CD2A4.";
string  melody4  =  "l4EEF8F4.GA2B2>C1.";
string melody = melody1 + "D2.p4" + melody1 + "D2p8A4." + melody2 + melody3 + melody2 + melody4;
s.add_channel(preamble  +  "o3"  +  bass);
s.add_channel(preamble  +  "o2"  +  bass);
s.add_channel(preamble  +  chords);
s.add_channel(preamble  +  melody);
s.add_channel(preamble  +  "o5v90"  +  melody);
s.write("song4.wav");
}

soundtrack object

This method will write a song to a wave file.

bool write(string filename)

Parameters:
filename
a string containing the desired file name for the wave file.

Return value:
true on success, false on failure.

Remarks:
Use the add_channel method first to add some music, otherwise you will create an empty wave file.

Example:
// Write a song.

#include  "soundtrack.bgt"

void  main()
{
soundtrack  s;
string  preamble  =  "t140u6";
string  chords1  =  "@1v97l2a8r100q12[C&E&G]2[<B&>E&G]2[<B-&>E&G]2[<A&>D&F]2<F+&A&B&>D<A-&B&>D&E<A&>C&E<G&>C&E<E&G&>C<D&F+&>C<F&G&>C&D<F&G&B&>D";
string  chords2  =  "[<B&>D&G]2[C&E&G]2q20<B4&>D4&B4C8&E8&>C8<<l4B&>D&B<A&>C&Aq12l2<G&B&>Gp8<G&B&>C&G<B&>D&G<B&>D&F<B&>D&G<G&>C&E";
string  chords3  =  "<G&>C&E<F+&>C&D<F&>C&D<F&B&>D";
string  chords4  =  "[<A&>C&F]3<G&B&>F<G&>C&El8q8r10<B-&>C&Gl4<A-&>C&Fl1<G&>C&Ep8l4<A-&>C&E<A-&>C&Dp4l1<E&G&>C";
string chords = chords1 + chords1 + chords2 + chords3 + chords2 + chords4;
string  bass1  =  "@3r100C2.CC2.CC2.CC2.C<B2>E2<A2.A>D2.D<G2.";
string  bass2  =  "E2.E<A2.A>F2.FE2<A2>F2<B2>E2<A2>";
string  bass3  =  "D2.D<G2.>F";
string  bass4  =  "D2.D4<G2.G>C1C1C4<G4p4C1";
string  bass  =  bass1  +  "G>"  +  bass1  +  ">F"  +  bass2  +  bass3  +  bass2  +  bass4;
string  melody1  =  "@2a10r10E2.p4E2.p4EE8FCD2.p4p8DD8E4.<B8>C4.DC<BA2p8>E.";
string  melody2  =  "l4G2.p4G2.p4B>C8<BAG2.p4p8B8B8>C8<B4.A8G4.>D8C4.<";
string  melody3  =  "E8E2.p8D8CD2A4.";
string  melody4  =  "l4EEF8F4.GA2B2>C1.";
string melody = melody1 + "D2.p4" + melody1 + "D2p8A4." + melody2 + melody3 + melody2 + melody4;
s.add_channel(preamble  +  "o3"  +  bass);
s.add_channel(preamble  +  "o2"  +  bass);
s.add_channel(preamble  +  chords);
s.add_channel(preamble  +  melody);
s.add_channel(preamble  +  "o5v90"  +  melody);
s.write("song4.wav");
}

soundtrack object

tone_synth synth
Gives direct access to the tone_synth object that the soundtrack object uses.

Keyboard Constants

These constants are meant to be used in conjunction with the key_down and key_pressed functions, in order to specify which
key to check for.

KEY_ESCAPE
KEY_1
KEY_2
KEY_3
KEY_4
KEY_5
KEY_6
KEY_7
KEY_8
KEY_9
KEY_0
KEY_MINUS - on main keyboard
KEY_EQUALS
KEY_BACK - backspace
KEY_TAB
KEY_Q
KEY_W
KEY_E
KEY_R
KEY_T
KEY_Y
KEY_U
KEY_I
KEY_O
KEY_P
KEY_LBRACKET
KEY_RBRACKET
KEY_RETURN - Enter on main keyboard
KEY_LCONTROL
KEY_A
KEY_S
KEY_D
KEY_F
KEY_G
KEY_H
KEY_J
KEY_K
KEY_L
KEY_SEMICOLON
KEY_APOSTROPHE
KEY_GRAVE - accent grave
KEY_LSHIFT
KEY_BACKSLASH
KEY_Z
KEY_X
KEY_C
KEY_V
KEY_B
KEY_N

KEY_M
KEY_COMMA
KEY_PERIOD - . on main keyboard
KEY_SLASH - / on main keyboard
KEY_RSHIFT
KEY_MULTIPLY - * on numeric keypad
KEY_LMENU - left Alt
KEY_SPACE
KEY_CAPITAL
KEY_F1
KEY_F2
KEY_F3
KEY_F4
KEY_F5
KEY_F6
KEY_F7
KEY_F8
KEY_F9
KEY_F10
KEY_NUMLOCK
KEY_SCROLL - Scroll Lock
KEY_NUMPAD7
KEY_NUMPAD8
KEY_NUMPAD9
KEY_SUBTRACT - on numeric keypad
KEY_NUMPAD4
KEY_NUMPAD5
KEY_NUMPAD6
KEY_ADD - + on numeric keypad
KEY_NUMPAD1
KEY_NUMPAD2
KEY_NUMPAD3
KEY_NUMPAD0
KEY_DECIMAL - . on numeric keypad
KEY_OEM_102 - <> or \| on RT 102-key keyboard (Non-U.S.)
KEY_F11
KEY_F12
KEY_F13 - (NEC PC98)
KEY_F14 - (NEC PC98)
KEY_F15 - (NEC PC98)
KEY_KANA - (Japanese keyboard)
KEY_ABNT_C1 - /? on Brazilian keyboard
KEY_CONVERT - (Japanese keyboard)
KEY_NOCONVERT - (Japanese keyboard)
KEY_YEN - (Japanese keyboard)
KEY_ABNT_C2 - Numpad . on Brazilian keyboard
KEY_NUMPADEQUALS - = on numeric keypad (NEC PC98)
KEY_PREVTRACK - Previous Track (KEY_CIRCUMFLEX on Japanese keyboard)
KEY_AT - (NEC PC98)
KEY_COLON - (NEC PC98)
KEY_UNDERLINE - (NEC PC98)
KEY_KANJI - (Japanese keyboard)
KEY_STOP - (NEC PC98)
KEY_AX - (Japan AX)
KEY_UNLABELED - (J3100)
KEY_NEXTTRACK - Next Track

KEY_NUMPADENTER - Enter on numeric keypad
KEY_RCONTROL
KEY_MUTE - Mute
KEY_CALCULATOR - Calculator
KEY_PLAYPAUSE - Play / Pause
KEY_MEDIASTOP - Media Stop
KEY_VOLUMEDOWN - Volume -
KEY_VOLUMEUP - Volume +
KEY_WEBHOME - Web home
KEY_NUMPADCOMMA - , on numeric keypad (NEC PC98)
KEY_DIVIDE - / on numeric keypad
KEY_SYSRQ
KEY_RMENU - right Alt
KEY_PAUSE - Pause
KEY_HOME - Home on arrow keypad
KEY_UP - UpArrow on arrow keypad
KEY_PRIOR - PgUp on arrow keypad
KEY_LEFT - LeftArrow on arrow keypad
KEY_RIGHT - RightArrow on arrow keypad
KEY_END - End on arrow keypad
KEY_DOWN - DownArrow on arrow keypad
KEY_NEXT - PgDn on arrow keypad
KEY_INSERT - Insert on arrow keypad
KEY_DELETE - Delete on arrow keypad
KEY_LWIN - Left Windows key
KEY_RWIN - Right Windows key
KEY_APPS - AppMenu key
KEY_POWER - System Power
KEY_SLEEP - System Sleep
KEY_WAKE - System Wake
KEY_WEBSEARCH - Web Search
KEY_WEBFAVORITES - Web Favorites
KEY_WEBREFRESH - Web Refresh
KEY_WEBSTOP - Web Stop
KEY_WEBFORWARD - Web Forward
KEY_WEBBACK - Web Back
KEY_MYCOMPUTER - My Computer
KEY_MAIL - Mail
KEY_MEDIASELECT - Media Select

Error Codes

These constants are the possible return values from the get_last_error function, which gives more specific information about any
errors that may have occured at runtime.

0 - Success.
-1 - Internal error.
-2 - One or more invalid parameters in last call.
-3 - Unable to initialize the audio subsystem.
-4 - Unable to initialize the keyboard subsystem.
-5 - File could not be opened for reading.
-6 - File could not be opened for writing.
-7 - Trying to load an unencrypted sound with force encryption enabled.
-8 - Encryption failed.
-9 - Decryption failed because the key was incorrect.
-10 - Decryption failed for an unknown reason.
-11 - No valid resource associated.
-12 - User chose to cancel.
-13 - The result was too large.
-14 - The supplied data could not be converted.
-15 - File not found.
-16 - File could not be read from pack file.
-17 - Out of memory.
-18 - Cannot go past the end of the file.
-19 - Invalid operation for this file mode.
-20 - Operation not supported.
-21 - No data available to read.
-22 - No data available to write.
-23 - Unable to initialize the text to speech subsystem.
-24 - Illogical operation.
-25 - Already initialized.
-26 - No callback function specified.
-27 - Unable to initialize the network subsystem.
-28 - No connected peer with this ID was found.
-29 - HTTP request error.
-30 - Directory not found.
-31 - Access denied.
-32 - Unable to initialize the mouse subsystem.
-33 - A sound device is already open.
-34 - Operation failed.
-35 - File already exists.
-36 - Unavailable in release builds.
-37 - Attempting to read badly formatted input.

Date and Time Constants

These constants can be used to retrieve the current local date and time on the user's system, which may or may not be accurate.
The information is updated in realtime, which means that the various constants will report different values as time passes during
the execution of the script.

DATE_YEAR - Integer.
DATE_MONTH - Integer (from 1 to 12).
DATE_MONTH_NAME - String (the English name of the current month).
DATE_DAY - Integer (from 1 to 31).
DATE_WEEKDAY - Integer (from 1 (Monday) to 7 (Sunday)).
DATE_WEEKDAY_NAME - String (the English name of the current weekday).
TIME_HOUR - Integer (from 0 (midnight) to 23 (11 in the evening)).
TIME_MINUTE - Integer (from 0 to 59).
TIME_SECOND - Integer (from 0 to 59).
TIME_SYSTEM_RUNNING_MILLISECONDS - Unsigned integer (specifies the number of milliseconds that the
operating system has been running for).
TIME_STAMP - Unsigned integer (specifies the number of seconds that have elapsed since midnight on January 1, 1970
which is also known as a Unix time stamp).

Script Properties

These are properties used to determine the current script information.

SCRIPT_COMPILED - Boolean (true if the script is a compiled executable, false otherwise).
SCRIPT_CURRENT_LINE - Integer (the line that is currently being executed, relative to the current script file).
script_current_file - String (the full path and filename of the script that is currently being executed). If the property is
retrieved from within a compiled game, the path is that of the source file on the original development machine.
SCRIPT_CURRENT_FUNCTION - String (the full declaration of the function that is currently being executed).
SCRIPT_EXECUTABLE - String (the full path of the script executable file. If the script is not compiled, this is the
location of bgt.exe).
COMMAND_LINE - String (the command line parameters given to a compiled BGT game executable. This is always
empty for non-compiled scripts).
BGT_VERSION - Double (the version of BGT that is being used to run the script.
BGT_REVISION - Integer (the revision relative to the current version of BGT that is being used to run the script.

Directory Constants

These string constants refer to special directories on the user's system. Not all constants are guaranteed to be available, in which
case they will simply return a blank string. All of the constants are guaranteed never to end with a backslash or slash character so
you should always append one if you wish to create a file in one of these directories for instance.

DIRECTORY_APPDATA - The file system directory that serves as a common repository for application-specific data.
A typical path is C:\Documents and Settings\username\Application Data.
DIRECTORY_COMMON_APPDATA - The file system directory that contains application data for all users. A typical
path is C:\Documents and Settings\All Users\Application Data. This folder is used for application data that is not user
specific. For example, an application can store a spell-check dictionary, a database of clip art, or a log file in this folder.
This information will not roam and is available to anyone using the computer.
DIRECTORY_DESKTOP_VIRTUAL - The virtual folder that represents the Windows desktop, the root of the
namespace.
DIRECTORY_DESKTOP - The file system directory used to physically store file objects on the desktop (not to be
confused with the desktop folder itself). A typical path is C:\Documents and Settings\username\Desktop.
DIRECTORY_LOCAL_APPDATA - The file system directory that serves as a data repository for local (nonroaming)
applications. A typical path is C:\Documents and Settings\username\Local Settings\Application Data.
DIRECTORY_MY_DOCUMENTS - The virtual folder that represents the My Documents desktop item.
DIRECTORY_TEMP - The directory where temporary files are stored.

Screen Reader Distribution

The screen reader related functions in BGT depend upon certain files being included with your application in order to operate.
This applies to System Access, and NVDA. Jaws for Windows and Window Eyes come with Active X COM components in
their installations that BGT calls upon to perform the requested tasks, which means that nothing needs to be distributed along
with your game in order to support these. Please note that older versions of Jaws for Windows do not come with this Active X
component. It is unclear at the time of this writing in which version the component was introduced, but our tests indicate that
Jaws version 10.0 and above work without problems. We will amend this section in the documentation as soon as we have more
specific information regarding exact version numbers.

For System Access and NVDA, BGT calls external dll's that have to be present. By default, the engine expects them to be in the
same folder as your game executable, though you can explicitly change the path with the screen_reader_set_library_path
function. You may find these dll files in the BGT installation directory/redist/screen_readers. The System Access dll has no
explicit license attached to it but is freely distributable, while the NVDA dll is distributed under the terms of the GNU Lesser
General Public license. This means that you need to include the file called license.txt along with NVDA's dll (found in the same
directory) if you wish to support this screen reader.

When calling the screen reader functions, they all take a parameter that indicates which screen reader to use. You may use the
following constants:

JAWS
WINDOW_EYES
SYSTEM_ACCESS
NVDA

BGT Command Line Options

When you choose an item in the BGT context menu by right clicking on a script file, what happens internally is that bgt.exe is
executed with a particular command line option. This means that by passing these options manually you can accomplish the same
tasks as the context menu does. Each option starts with a - (dash) character. The following command line options are available:

Executing a script

You may execute a script simply by passing its name in quotes. No special options need to be specified.

bgt.exe "path/to/my/script.bgt"

Compiling a script

The -c and -C commands are used to compile scripts. -c compiles a debug build, and -C compiles a release build. If only -c or -
C is specified, a dialog asking for a script to compile will appear. Optionally you may include a filename within quotes after the
command, which will result in that script being compiled directly.

bgt.exe -c
Shows a dialog box asking for a script to compile, and makes a debug build of it.

bgt.exe -C
Shows a dialog box asking for a script to compile, and makes a release build of it.

bgt.exe -c"path/to/my/script.bgt"
Compiles a debug build of the specified script.

bgt.exe -C"path/to/my/script.bgt"
Compiles a release build of the specified script.

Launching the registration manager

You may launch the registration manager by passing the -r command.

bgt.exe -r

Debug Versus Release Builds

When you compile a BGT script, you can choose whether to compile it as a release or as a debug build. Release builds are
different from debug builds in the following ways:

1. Debug builds store a lot of additional information about your script such as the names of the files on your system that
the code is contained in, line numbers, etc. This is good for debugging as the engine is able to provide you with very
detailed information when an error occurs, but is not a good thing when you release your work to the public because more
details about your code can be deciphered. The actual source code that you wrote is not stored anywhere, but it is a little
easier for a person with malicious intentions to extract more information based on this debugging data. Release builds
therefore exclude this information, resulting in a smaller executable that is harder to reverse engineer. However, debugging
with a release build is close to impossible because so much vital information is missing. In other words you should use
debug builds while you are testing, and release builds when it is time to ship your game to the masses.
2. Debug builds contain additional code to make the results from the script profiler significantly more accurate. Release
builds exclude this, resulting in a performance increase but rendering the profiler almost useless. Thus, you should use
debug builds when you want to profile your code and release builds when you want the extra speed.

When you run a BGT script from source, it is automatically run in debug mode.

IMPORTANT: PLEASE READ THE TERMS AND CONDITIONS OF THIS LICENSE AGREEMENT CAREFULLY
BEFORE INSTALLING AND/OR IN ANY WAY USING THIS SOFTWARE PRODUCT:

Copyright (C) 2010-2014 Blastbay Studios

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any
damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and
redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this
software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any distribution of the software, but does not have to be present in derivative
works produced with the software (see clause 1 above).

Blastbay Studios

Blastbay Studios values your privacy, and collects very little personal information from you. The only information that is ever
stored in our files is the limited amount of data given to us when an order is placed which includes but is not limited to your name,
and email address. No financial information is ever obtained or stored by us.

Under no circumstances will any of your private information be sold, rented, sublicensed or in any way shared with a third party
without your prior written consent.

We wish to thank the following people for their respective contributions:

The Angel Script Project
AngelScript is the scripting interpreter used in BGT, and is what gives the engine its infinite flexibility and power while still
maintaining language simplicity.

The ENet Project
ENet is a rock solid layer on top of the UDP protocol, optimized especially for high-speed games. This is what BGT uses to
provide its networking capabilities.
Note: For licensing information related to the ENet library, please see the website linked to above.

Damien C. Pendleton
A huge thanks goes to Damien Pendleton for co-writing as well as maintaining the ever expanding BGT documentation, as well
as writing some include scripts and a great number of the string related functions found in the engine.

Felix Grützmacher
We also wish to express a big thanks to Felix Grützmacher for his fantastic series of tutorials on how to create games from the
ground up, his pathfinding tutorial, as well as for his creative ideas and suggestions during the long months of development.

We would also like to extend our thanks to our infinitely patient and skilful beta team members (Damien Pendleton, Lukás
Hosnedl, Oriol Gómez and Joshua Duncan) for helping to see this project all the way to the end and providing a constant stream
of invaluable feedback.

We also wish to thank Bob Barnes for his fabulous work narrating the audio version of the BGT language tutorial.

And last but not least, thanks to all of you who contribute ideas and suggestions as well as provide help to others on the Blastbay
discussion forum. No software can survive without a user base and community, and we wish to express our great appreciation
for all that you, the end users, are contributing.

The BGT engine is in constant development. The following is a list of the current limitations that we intend to address in future
releases.

The network object is too complicated. We need a higher level wrapper around it that simplifies common tasks.

Version 1.3 (revision 2):
* Re-released the engine as freeware. All features that used to be exclusive to the paid versions of the engine are now available
for free. Enjoy!
* Removed the "check for updates" feature. New releases will be announced on the Blastbay website as well as in the forum,
which has an rss feed.
* Added the ability to specify a custom port number as part of an http request (sponsored by Valiant Galaxy Associates and
Alter Aeon).
* Fixed a very serious problem in the audio subsystem that would cause random crashes and hangs.
* Set an upper bound of 10000 by 10000 for the size of a pathfinder map.
* Made the calendar object verify that the parameters given to the set method are in range, as wildly incorrect arguments could
actually result in crashes.
* Fixed a crash that would occur in the tone synth due to incorrectly calculated sample offsets when calling rewind or rewind_ms
(thanks Corey).
* The max_distance property in the sound_pool include class is now sound specific, which means that different sounds can have
different max distances rather than just one global setting (thanks Aaron).
* Fixed a runtime error that would sometimes occur in the audio_form include class when using multiline input fields (thanks
Aaron).
* Updated the language tutorial with some information about uninitialized variable usage (thanks Aaron).
* Fixed some ambiguities in the documentation for the call method of the library class.

Version 1.3 (revision 1):
* Updated the script interpreter to the latest version which fixes some more bugs found by users, and adds the following features:
1. The infamous "unexpected end of file" compilation error now shows a lot more information.
2. It is now possible to declare several class properties of the same type on a single line, separated by comma.
3. It is also possible to initialize class properties in their declaration, rather than having to do it in the constructor.
Note: This means that some obscure scripts, where a child class calls a method in the parent class that in turn tries to access its
members before its constructor has been called by the child class, will now result in a null pointer access script runtime error.
This happens very rarely, but should be taken into account if you are upgrading and have scripts that may potentially reproduce
the above scenario.
4. Large scripts will now compile significantly faster, and loading of precompiled byte code is faster as well.
5. The script compiler no longer implements a default constructor for classes that define a constructor with arguments. If you
want both a default constructor and a constructor that takes arguments, you must define both of them explicitly (see the language
tutorial for more information).
* Added DLL call support.
* Added the ability to compile scripts either as debug or release builds, and added documentation to explain the differences (see
appendix h).
* Added some basic serialization functions that make it easier to save and load data from files or from memory.
* Made it possible to serialize and restore the exact state of the random number generator at any point in time.
* Added an object called combination which contains algorithms to calculate different types of combinations from an arbitrarily
large set of items (sponsored by Marc Andersen).
* Added a pan property to the tone synth object which affects all subsequently generated notes.
* Added reverberation to the tone synth object, which is controlled by various properties.
* Added a new function called is_admin which checks if the program is being run with administrator priviliges (thanks Liam).
* Added some new functions to force/simulate keystrokes (thanks Nikola).
* Added two new functions (string_base64_encode and string_base64_decode) which makes it possible to convert binary data
to and from a printable, Ascii compatible representation.
* The keyhook is now automatically uninstalled when the user leaves the game window, and installed again when the window is
reactivated. This is to prevent lag in certain screen readers when the user is working with an application other than the game.
* Upgraded to the Visual Studio 2010 compiler, which gives a performance boost.
* Changed it so that initial text that is placed in an input box is selected automatically when the window appears (thanks Nikola).
* Added a new folder constant (DIRECTORY_MY_DOCUMENTS), see appendix e for more details (thanks Liam).
* Significantly optimized the way audio files are read, resulting in a great performance improvement when loading sounds (thanks
Aaron).
* Optimized the string_contains function.

* Significantly optimized the internals of the sound_pool include class so that it now runs much faster.
* Made the script compilation result dialog show the amount of time that the compilation took.
* Added a speak_to_file method to the tts_voice object (thanks Marc).
* Added a function to hide the game window (thanks Nikola).
* Added the concept of named items to the dynamic_menu include class, making it easier to figure out what the chosen item is
regardless of in what order the items were added.
* Added a new method to the audio_form class that allows a listbox cursor to be changed (Thanks Damien).
* Allowed the home and end keys to navigate to the top and bottom of a listbox in an audio form.
* Fixed a serious bug in the audio subsystem which would cause random crashes.
* Fixed a crash which would occur when converting large floating point numbers to strings (thanks Corey).
* Fixed a bug which would cause the string_split function to enter an infinite loop if the NULL terminator was used in the
delimiter (thanks Damien).
* Fixed a crash which would sometimes occur after a runtime error (thanks Corey).
* Improved support for the Window Eyes screen reader (thanks Aaron and Jason).
* Fixed a problem where erroneous warning messages could be triggered if the initialization of the global variables failed (thanks
Damien).
* Fixed a bug in the sound_pool include class which made it try to reload a sound that came back within earshot, from the
currently used storage rather than the storage that was active when the sound was initially loaded (thanks Liam).
* Fixed a bug in the dynamic_menu include class which would cause the intro message not to be heard if the auto_speak_first
parameter was true and the first item used an output different from the intro message (thanks Aaron).
* Fixed a bug in the audio_form include class in the get_list_count method where the maximum index was being returned rather
than the number of items as described in the documentation (Thanks Damien).
* Fixed several multiline input field bugs in the audio_form include class (Thanks Damien).
* Updated the language tutorial with information about the new behavior of automatic default constructors, and about how to
assign values to class properties directly in their declaration.
* Added a serialization tutorial.
* Documented the string object.
* Added four missing overloaded operators to the list in the language tutorial (opPreInc, opPreDec, opPostInc and opPostDec).
* Added a note to the number_to_hex_string documentation regarding the fact that it produces lowercase results (thanks
Jordan).
* Added a list of command line options available in the BGT engine (see appendix g).
* Updated the documentation for the profiler to explain the differences when profiling debug versus release builds.
* Failed to document the new array methods mentioned in the change log for version 1.2 revision 1 (thanks Marc).
* Failed to document the new get_size method mentioned in the change log for version 1.2 revision 1 (thanks Damien).
* Failed to document the sound_is_playing method in the sound_pool include class(Thanks Lukás).
* Failed to document several methods of the dynamic_menu class (Thanks Lukás).
* Fixed an error in one of the examples in the language tutorial to do with integer calculations.
* Fixed a few typos in the html code in the documentation (thanks Nikola).

Version 1.2 (revision 1):
* Modified the internal encryption key scheme to be considerably more secure.
IMPORTANT! The encryption functions are not backwards compatible. You will not be able to decrypt any data encrypted
with earlier revisions of BGT, using this version. These measures were taken to make it harder for hackers to steal keys from
memory dumps.
* Updated the script interpreter to the latest version which fixes some more bugs found by users, and adds the following features:
1. A method called reserve to the array object which is useful for optimizing performance because it preallocates memory for
future elements.
2. A method called is_empty to the string, the array and the dictionary objects which sometimes is a more efficient shortcut than
checking the length.
3. A method called get_size to the string, the array and the dictionary objects. In the two former cases, get_size is the same as
the length method.
4. Some great optimizations to the array methods that call user defined functions internally, such as find and sort_ascending/
sort_descending.
* Added joystick support.
* Added some new functions to perform script code profiling.
* Added an instance object that allows you to ensure that only one copy of your game is running (thanks Damien).

* Added a function called set_error_output which enables you to have internal engine errors automatically printed to a file
throughout the execution of the program, along with some other information such as the script call stack when the error
happened.
* Added a method called get_headers to the http object that retrieves only the headers from a given location on an http server
(thanks Nikola).
* Added two functions (set_sound_master_volume and get_sound_master_volume) to adjust the volume of the entire sound
output in a given game.
* Added two functions (screen_reader_set_library_path and screen_reader_unload_library) which provide you with more
control over how BGT uses the dll files for System Access and NVDA (sponsored by Lukáš Hosnedl).
* Added an optional parameter to the input_box function to fill the box with some initial text (thanks Claudio).
* Added a global script property called SCRIPT_EXECUTABLE which retrieves the full path to the program (sponsored by
Lukáš Hosnedl).
* The set_sound_decryption_key method now opens the default audio device implicitly if a device isn't already open.
* Reduced the size of precompiled game executables a little more.
* The maximum stack size has been increased from 2000 to 10000 functions.
* Updated the performance optimization tutorial with a section on reserving memory in arrays.
* Added support for progress bars with beeps to the audio_form include class (thanks Christopher).
* Added an optional parameter to the non-extended play methods in the sound_pool include class, to allow these to create
persistent sounds as well (thanks Ryan).
* Added support for multiselection enabled listboxes in the audio_form include class (Thanks Damien).
* Added several new methods and constants for retrieving, setting and managing control information in the audio_form include
class (Thanks Damien).
* Added support for additional keys such as right control and NumPad Enter to be used in the audio_form class (Thanks
Damien).
* Improved several speech and keyboard behaviors in the audio_form class (Thanks Damien).
* Changed a behavior in the audio_form so that now only one progress bar in the form may be flagged as global (Thanks
Damien).
* Changed the name of the redist/screen-readers directory to redist/screen_readers, as it was originally supposed to be.
* Fixed a crash in the load_from_memory method of the sound object if invalid input is given (thanks Nikola).
* Fixed a bug that would cause sound cloning to be performed incorrectly if a file with the same name was loaded both from a
pack file and from disk.
* Fixed a bug in the write method of the logger class where a file would still be written to even if the log was empty (Thanks
Damien).
* Fixed a bug in the logger class that caused line breaks not to be written correctly (Thanks Damien).
* Fixed several crashes in the audio_form class (Thanks Damien).
* Fixed a bug in the audio_form include class where the cursor of a listbox would reset if adding an item to a list (Thanks
Damien).
* fixed a bug in the audio_form class whereby the set_tts_object method would fail if the form was active (Thanks Damien).
* Fixed a bug in the audio_form class that pasted multiline text into a single line field (Thanks Damien).
* Notice: Failed to document the additional parameters in the create_button method in the audio_form include class (thanks
Damien).
* Fixed a few typos in the language tutorial (thanks Marc and Michael).

Version 1.1 (revision 5):
* Updated the script interpreter to the latest version which fixes some more bugs found by users, and adds the following features:
1. Rather than crashing when there isn't enough memory available to allocate an array, the engine now triggers a regular script
runtime error.
2. The script compiler is now separate from the virtual machine, which means that compiled game executables are quite a bit
smaller than before.
* Added the ability to store a pack file inside precompiled executables that can be accessed both by the pack_file object and by
the set_sound_storage function (sponsored by Marc Andersen).
* Added four new functions (key_up, key_released, keys_up and keys_released).
* Added a method called get_keys to the dictionary.
* Added a method called load_from_memory to the sound object which accepts a buffer of data, rather than a filename.
* Added a method called write_wave_sound to the tone synth object which mixes its data down into a sound object rather than
to a wave file.

* Significantly optimized the internal handling of string constants which will give a performance improvement of about 260 % for
scripts that use a lot of literals.
* Optimized the dictionary to use a hash map implementation from the Boost C++ libraries rather than the standard map, which
increases performance a bit.
* Optimized the string_contains function slightly.
* Added a new function called garbage_collect which allows you to control the internal garbage collector used in BGT a little
more precisely.
* Added a new function called file_copy.
* Made the argument to the read method in the file object optional.
* Decreased the number of internal memory allocations that take place when a timer object is created, which gives a slight
performance boost.
* Added a new boolean property to the dynamic_menu include class, enable_home_and_end, which enables and disables these
extra navigational keys to jump to the start and the end of the menu respectively (thanks Damien).
* Optimized the dynamic_menu include class so that it now only initializes Sapi when it is actually needed (thanks Marc).
* Added some safeguards to catch C++ exceptions during execution and report them with a script runtime error, rather than
simply crashing.
* Scripts are now compiled with fewer suspend instructions which increases over-all execution performance.
* Removed the limiter from the mixing chain of the tone synth. A simple peak normalize function is all that's needed to prevent
clipping.
* Added a tutorial on how to include a pack file in the executable.
* Added a tutorial on how to optimize the overall performance of BGT games.
* Updated the soundtrack include class to expose the support for triangle waveforms (thanks Felix).
* Fixed a bug where the engine would crash if the user tried to convert an infinite or NaN floating point number into a string
(thanks Emanuel and Andreas).
* Fixed a bug where the window event queue wasn't cleared before a question dialog box appeared, which could cause issues
with the keyboard in certain cases (thanks Christopher).
* Fixed some spelling errors in the documentation, and updated the change log to indicate which features were sponsored by
third parties both in the past and for this release.
* Cleaned up the code in the dynamic_menu include class a great deal, removing a lot of unnecessary parts.
* Changed the BGT version number in the installer from 1.0 to 1.1 (thanks Kevin).

Version 1.1 (revision 4):
* Updated the script interpreter to the latest version which fixes some more bugs found by users.
* Added a pitch property to the tts_voice object.
* Added two new global constants (TIME_SYSTEM_RUNNING_MILLISECONDS, and TIME_STAMP).
* Reworked the script compilation process to make the resulting game executables smaller and more compact.
* Changed the numeric properties from doubles to integers in the tts_voice object to save space and some performance, as
doubles served no purpose.
* Changed the dynamic_menu include class so that the methods that add items return the menu position of the new item, rather
than a boolean (thanks Damien).
* Fixed a serious memory leak that could consume several hundred mb in a matter of seconds if the speaking property of the
tts_voice object was accessed often.
* Fixed a problem where the garbage collector could overflow if an engine function that returned an array was called too often in
a tight loop without a wait call (thanks Nikola).
* Fixed a bug where the return values from the diff_hours, diff_minutes and diff_seconds methods in the calendar object would
be wrong if the years of the time stamps were different.
* Fixed some artifacts in the tone generator (thanks Jayson and Damien).
* Fixed the handling of global variable initialization errors in compiled game executables (thanks Damien).

Version 1.1 (revision 3):
* Limited the number of values that can be written to the registry for a given game to 50 (thanks Willem).
* Fixed a runtime error in the dynamic menu that would happen when Sapi was used with no explicit tts_voice handle (thanks
John).
* Fixed a bug in the run_extended method in the dynamic menu include class when it was used with screen readers (thanks
Hayden).

Version 1.1 (revision 2):
* Added a settings object which allows you to read and write in the Windows registry.
* Added two global constants (double BGT_VERSION and int BGT_REVISION) to determine the version and revision of the
engine that is being used (thanks John).
* Added the ability to set a speech output mode (either Sapi 5.1 or a screen reader) to the dynamic_menu include class.
* Notice: Failed to mention that compiled scripts can now read the command line parameters given to them, in the change log for
version 1.1 revision 1.
* Notice: Failed to document the set_speech_mode method in the audio_form include class in version 1.1 revision 1 (thanks
Damien).
* Fixed a bug that caused the result from the get_characters function to be slightly out of date (thanks John).
* Fixed a bug where the update_sound_start_values method of the sound_pool include class didn't work for stationary sounds
(thanks Damien).
* Fixed a bug that caused the index operator for strings not to retrieve characters properly.

Version 1.1 (revision 1):
* Updated the script interpreter to the latest version which fixes some more bugs found by users, gives significant performance
boosts in many areas, and adds the following features:
1. Declaring functions with default arguments.
2. Several new features to the array object such as sorting, reversing, searching etc.
* Added support for mouse devices.
* Added the pathfinder object to find intelligent paths between two points with a lot of customization options.
* Included one of the add-ons for the script interpreter, a vector object to represent locations in space with many overloaded
operators for easy manipulation.
* Added support for communicating directly with four popular screen readers in order to have text spoken.
* Changed the memory management functions from the standard malloc and free to custom replacements which are much faster,
giving a great performance boost of the entire engine as large amounts of dynamic memory are allocated all the time during
execution.
* Optimized all functions and methods that return objects (particularly arrays), resulting in a significant speed increase for these.
* Added two functions called keys_down and keys_pressed to return a list of all the keys that are currently down or that have
just been pressed.
* Added a function called list_sound_devices, and another called open_sound device which enables the opening of any given
device on the user's system rather than just the default one (thanks Liam).
* Added a function to read environment variables (thanks Marc).
* Added a function to read text from the clipboard.
* Added an option to the BGT program group to check for engine updates.
* Updated the ENet library to the latest version which fixes some bugs and improves performance even further when using
reliable packets.
* Changed the behavior when a runtime error occurs, so that it is now possible to choose whether or not the complete stack
trace text should be copied to the clipboard before the program exits.
* Added a tutorial about pathfinding (thanks Felix).
* Extended the language tutorial with a description of functions with default arguments.
* Made the automatic window polling occur a little less frequently which gives a slight performance boost.
* Added the install_keyhook and uninstall_keyhook functions to enable BGT games to access the keyboard even while the Jaws
for Windows screen reader is active.
* Added a method called read_until to the file object that reads data until a certain criteria has been met.
* Added a method called force to the timer object to simulate that a certain amount of time has elapsed, which is needed for
serialization.
* Filtered out the signed/unsigned mismatch warning from the compiler output as it was being triggered in the most trivial cases.
* Significantly improved the performance of the keyboard management functions.
* Completely rewrote the generate_computer_id function to be a lot more reliable.
IMPORTANT! The function is not backwards compatible with earlier versions, as they had serious flaws. Thus, none of the
generated ID's will be the same.
* Rewrote the implementation of the tts_voice object so that it now uses the IDispatch COM interface, which allows for more
functionality such as enumerating and changing voices.
* Optimized the stop method of the tts-voice object slightly.
* Improved the unpredictability of the seed to the random number generator so that it no longer generates the same sequence of

numbers if two BGT instances are launched within the same second.
* Added a function called string_distance to check how similar two arbitrary strings are.
* Added checks for invalid input to the key_down and key_pressed functions as this would otherwise cause a hard crash.
* Added the ability to generate triangle waves to the tone synth.
* Changed the key_down and key_pressed functions to take an integer as their parameter.
* Added support for screen readers to the audio form include class (thanks Damien).
* Added a new optional parameter to the extended methods in the sound_pool include class to set whether or not a returned slot
should be persistent and not get automatically cleaned up after playback is finished.
* Extended the FAQ with a brief description of how to protect your sounds with encryption in a way that makes it harder to
break.
* Added some new information to the documentation about how to use the sound_pool include class properly.
* Fixed a number of serious crashes that would occur occasionally on Windows 7 (thanks Lukáš and Daniel).
* Fixed an ancient and very serious bug that caused sounds to mysteriously inherit properties from each other (thanks Lukáš,
Michael and Damien).
* Fixed a memory leak that would occur on exit.
* Fixed a bug in the network_event overloaded assignment operator that would cause it to crash seemingly at random (thanks
Emanuel).
* Fixed a bug in the tone generator that would cause it to produce notes that were slightly off key.
* Fixed a bug where the properties SCRIPT_CURRENT_LINE, SCRIPT_CURRENT_FILE and
SCRIPT_CURRENT_FUNCTION were returning error values if accessed before main began executing.
* Fixed a number of bugs in the audio form include class (thanks Damien).
* Fixed a bug in the sound_pool include class where more sounds would start playing after a pause/resume operation.
* Fixed a bug where some uncommon international characters would not be spoken properly by the tts-voice object (thanks
Lukáš).
* Fixed a bug in the file object where the reached_end property would not be accurately set when reading files in binary rather
than text format (thanks Damien).
* Fixed a bug where the last error flag would not be correctly set after a call to one of the encryption or decryption functions
(thanks Damien).
* Fixed a bug in the input_box function where it would freeze the window if no information text was given (thanks Nikola).
* Fixed a bug where the wrong error code would be given if a file could not be opened for reading in binary mode using the file
object.
* Fixed some errors in the documentation layout (thanks Damien).

Version 1.0:
* Updated the script interpreter to the latest version which fixes some more bugs found by users, as well as adds the following
features:
1. The ability to convert booleans to strings.
2. Four new methods to the array called insert_at, remove_at, insert_last and remove_last.
3. Explaining messages when failing to initialize a global variable after a compilation.
4. Explaining messages when trying to declare variables within a switch/case statement.
5. The ability to write floating point numbers as .42, without the leading 0.
* Updated the ENet library to the latest version which improves bandwidth throttling of reliable packets.
* Added a calendar object for more advanced date and time calculations.
* Added the pack_file object to enable reading and writing pack files, which are also used by the sound object.
* Extended the language tutorial with several new chapters as well as a section about array initialization lists (thanks Felix).
* Extended the language tutorial with a description on how to use multidimensional arrays.
* Added a series of comprehensive tutorials that explain how games are written in practise (thanks Felix).
* Added support for sending synchronous HTTP GET and POST requests through two simple functions.
* Added the http object which allows for more advanced HTTP access using an asynchronous interface, which means that any
number of files/resources can be downloaded in the background from the Internet.
* Added two functions (url_encode and url_decode) to make it easier to assemble parameters for Get and Post HTTP requests.
* Added the generate_computer_id function.
* Added a function to run third party executables.
* Added several new file and directory functions (directory_create, directory_delete, directory_exists, and file_delete).
* Added an article to the tutorials section of this documentation that explains the various methods used for game registration.
* It is now possible to modify an individual character at a particular location in a string, just as if the string were an array.

* Added two hash functions to the engine (file_hash and string_hash) which expose sha256 and sha512 hash generation at
present, returned in either hex or binary form.
* Added a number of constants listed in appendix e that contain the paths of special folders on the system (thanks Liam).
* Added the get_last_error_text function that converts the value returned by get_last_error to its corresponding textual
description (thanks Damien).
* Added a function to check whether the game window is currently active.
* Added the COMMAND_LINE property (listed in appendix d) to retrieve the parameters given to a compiled game
executable.
* The array object is now fully documented (thanks Damien).
* Added the reset method for the dynamic menu include class to the documentation (thanks Lukáš).
* Added a method called set_sound_object to the dynamic_menu include class which allows you to specify a sound object that
should be used for auditory feedback.
* Added a virtual audio window include class (thanks Damien).
* Added the soundtrack include class which wraps the tone_synth object in a simpler interface (thanks Felix).
* The error dialog now displays the contents of a faulty line as well as its number, rather than just the number on its own.
* Changed the behavior of the file_exists function so that it only works with files, and added the directory_exists function instead.
* Removed every trace of the MD5 hashing algorithm from the encryption layer in the engine, which is not only good for
improving security but also for humanity at large.
Please note: All data that was encrypted with a prior version of BGT will no longer be valid and cannot be decrypted with this
release.
* Fixed a bug where I had accidentally made float equal to double in the engine.
* Fixed a bug in the number_to_words function (thanks Daniel).
* Fixed a bug in the string_trim_left function that would cause it to trim one character too little.
* Fixed a bug in the string_contains function that would cause it to erroneously return a positive result even if a particular
occurrence was not found in rare cases.
* Fixed a mistake in the documentation for the set method in the dictionary object (thanks Damien).
* Fixed a typo in the example for the freq_ms method in the tone_synth object (thanks Jason).
* Updated the end user lisence agreement to clarify the fact that an unlimited number of non-commercial games may also be
created with the pro single version of the engine.
* Fixed a typo in the documentation for the write method of the logger include class (thanks Michael).
* Fixed a large number of trivial typos in the documentation.
* Added the default start and end time values for the edge fades in the tone_synth object to the documentation (thanks Oriol).

Beta Version 0.5.1:
* Updated the end user license agreement to cover the difference between the pro single and pro unlimited versions of the
engine.

Beta Version 0.5:
* Updated the script interpreter to the latest version which fixes some more bugs found by users.
* Added a powerful sound management include class that handles positioning of an arbitrary number of sound sources in a 1d or
2d environment.
* Updated the functions in sound_positioning.bgt to be more flexible, and added two new ones with extra parameters. This will
require some minor modifications to scripts that currently make use of position_sound_2d.
* Added support for high performance networking, optimized especially for fast paced action games.
* Added a method called run_extended to the dynamic menu include class which allows you to specify a starting position for the
menu cursor, and to have the initially selected item automatically announced after the intro finishes.
* Added a reset method to the dynamic menu include class that allows you to remove all previously added items, as well as
optionally reset all the behavior properties to their defaults.
* Added a boolean property called paused to the sound object which is set to true if a sound has been paused with an explicit
call to the pause method.
* Added a function to convert numbers to their corresponding hexadecimal string counterparts.
* Added two functions (find_files and find_directories) that return a list of all the files and all the directories in a given path,
respectively.
* Made the engine registerable, with automatic detection of the purchased version (lite or pro) based on the key given.
* Added the ability to compile scripts into standalone executables if a lite or pro license has been purchased.
* Added a credits section to the documentation.

* Added a function called get_call_stack_size to retrieve the number of functions currently on the stack.
* Allowed the use of multiline strings (e.g. string literals that contain a line feed) (thanks Nick).
* Restructured the script execution flow, which will allow the engine to invoke user defined callback functions in the future.
* changed the range of the rate property in the tts_voice object from -100 (slowest) to 100 (fastest), so that it is now -10
(slowest) to 10 (fastest) which prevents situations where no change is perceived even though the property is actually modified
(thanks Jason).
* Fixed a slight inconvenience in the dynamic menu include class where the text to speech and the audio options were not
cooperating with one another.
* Fixed a bug in the random number generator that would cause it to generate the same sequence of numbers every time (thanks
Sorin).
* Changed the internal buffer size for streaming sounds in order to boost performance slightly.
* Fixed a bug that would sometimes cause static sounds to have an initial starting position greater than 0.
* Improved the error output when including script files that do not exist, so that the engine now displays all the locations in which
it tried to find the file rather than just the last one.
* Fixed a bug in the #include searching algorithm so that it now derives the current base directory for a relative include path
based on the location of the script that is trying to include it (thanks Damien).
* Fixed a few bugs that would cause crashes when calling certain functions during global variable initialization (thanks Damien).
* Made the exit function usable even during global variable initialization.
* Fixed a bug where the volume property in the tts_voice object would get incorrect values (thanks Jason).
* Fixed a bug where the key_down function could erroneously report that a key was down if the user left the window while the
key was still down but released later.
* Fixed a bug where certain special characters would not be spoken properly by the tts_voice object (thanks Oriol).
* Fixed a few typos in the language tutorial.

Beta Version 0.4.1:
* Changed the number_speaker class back to only have one argument in the speak and speak_wait functions, and have the
include_and parameter as a property instead. This means that code written with beta versions earlier than 0.4 can remain
unchanged.

Beta Version 0.4:
* Changed the way that the script engine interprets characters, so that it now works exclusively with the extended Ascii character
set and no longer complains about invalid Unicode sequences every now and then.
* Added two functions (ascii_to_character and character_to_ascii) to convert between string characters and their numerical
Ascii representations.
* Added two functions (hex_to_string and string_to_hex) to convert between normal strings and their hexadecimal
representations (thanks Felix).
* Added a function called get_characters which retrieves actual characters pressed by the user in the window, so that one no
longer has to rely on the BGT keyboard handling to intercept written text.
* Added functions to encrypt and decrypt files and strings using an algorithm that has not yet been cracked by anyone.
* Added an assert function to aid in debugging.
* Added a function called get_call_stack that retrieves a string with the entire nested stack of function calls leading up to the
current execution point.
* Added 15 new math functions (round, absolute, floor, ceiling, sine, cosine, tangent, arc_sine, arc_cosine, arc_tangent,
square_root, exponent, log, log10, and powwer).
* Updated the number_to_words function to include an extra boolean parameter which specifies whether the word "and" should
be inserted in the appropriate places in the output string.
Please note: Existing code using this function needs to be slightly modified in order to run.
* Added a property called running to the timer object that specifies whether a particular timer is running or paused (thanks
Damien).
* Limited the call stack size to 2000, which means that the engine will now prevent endless function recursion conditions etc.
* Added four global script properties (SCRIPT_COMPILED, SCRIPT_CURRENT_FILE, SCRIPT_CURRENT_LINE, and
SCRIPT_CURRENT_FUNCTION).
* Added an include class that makes it easy to print log files with dated entries (thanks Damien).
* Added an include function to print out properly formatted currencies (thanks Damien).
* Modified the number_speaker class to reflect the change in the number_to_words function.
Please note: Existing code using this class needs to be slightly modified in order to run.

* Changed the internal random number generation algorithm to one that runs considerably faster, supports a wider range of
values and is more statistically accurate.
* Added some more details to a rather uninformative engine error message.
* The help file now has a complete searchable index (thanks Damien).
* Fixed a memory access violation that would cause a hard crash when the exit function was called (thanks Lukáš).
* Fixed a bug in the string_replace function that would sometimes cause inaccurate replacements (thanks Damien).
* Fixed a bug where script files would be included incorrectly on Windows 7 64 bit machines (thanks Jodie).
* Fixed a bug where the engine would crash if presented with an unsupported wave file format other than PCM (thanks Casey).

Beta Version 0.3.2:
* Fixed some mistakes in the documentation where certain files were not being included.
* Added 7 more string functions to convert strings between upper and lower case, and to check whether a string meets certain
conditions such as if it's all alphabetic or numeric characters.
* Modified the license agreement to properly state the versions of Windows on which the engine can be run.

Beta Version 0.3.1:
* Changed the name of the function add_tts_item to it's proper name add_item_tts which is referenced in the documentation.
* Removed some debugging message boxes from the sound object that had managed to sneak in before release.

Beta Version 0.3:
* Updated the script interpreter to the latest version which fixes some more bugs found by users.
* Added global date and time properties.
* Reworked the menu class to support Sapi.
Please note: Existing code using this class needs to be slightly modified in order to run.
* Added a file_exists function.
* Added two more string functions (string_contains and string_replace) (thanks Damien).
* Added the ability to compile scripts into standalone executables, however this functionality is only available to preorder users.
* Modified the end user license agreement to reflect the fact that demo users will not actually be allowed to compile binary
executables.
* Added a privacy policy.
* Some bugs that would occur when playing audio on x64 machines have been fixed.
* The pitch of a sound can no longer go quite as low as in previous versions, as it would often cause unstable behavior.
* Added a property to the sound object called "pitch_lower_limit" that allows you to check how far down the pitch can go for
the current sound.
* Disabled the system menu that would appear when pressing the alt key in the game window, as this interfered with the BGT
keyboard input (thanks Lukáš).
* The engine now looks for include files properly, e.g. first in the current working directory and then in the BGT include
directory.
* Changed the compiler output to specify the full path of the file that the error or warning message in question is refering to. The
word "section" has also been changed to "file".
* Added some extended error checking to the compiler which is useful in some rare cases.
* Moved the Blastbay office to a new location (thanks Stockholm furniture Removal Services, sort of).

Beta Version 0.2:
* Updated the script interpreter to the latest version which fixes some more bugs found by users.
* File reading and writing support has been added, both for binary and text files.
* Properties to get the current position and total length of a sound have been added to the sound object, along with a method to
set the position as well.
* Partial support for Microsoft Sapi 5.1 text to speech has been added.
* A fully featured waveform tone generator has been added which allows the creation of musical compositions using sine,
sawtooth and square waves (sponsored by Damien Pendleton).
* Fixed a few minor errors in the tutorial.
* The error edit control now uses word wrap rather than a scroll bar.
* Switched from the MinGw C++ compiler to Visual Studio 2008, which brings us up to date with all the latest Microsoft
features, and which also produces an executable that is roughly 400 kb smaller and runs considerably faster.
* Fixed a few object methods where I had forgotten to set the global error flag based on the results.

Beta Version 0.1:
* Fixed a few errors in the tutorial.
* Added an Edit Script option to the context menu for BGT scripts.
* Added a list of current limitations to the help file.

Alpha version 0.5:
* Updated the script interpreter to the latest version which fixes some more bugs related to loops and object properties.
* A bunch of new functions to manipulate strings have been added (thanks Damien).
* A function to convert a string to a number has been added.
* Two new functions have been added to the include directory which help to simplify the positioning of sounds in 1d and 2d
environments.
* The set_sound_storage function has been added, making it very easy to read sounds from pack files as soon as the packing
features themselves are implemented.
* Enabled automatic scrolling for the edit field where errors and warnings are shown.
* Finally fixed the pitch bug properly after it came up once again in an even more weird form (thanks Lukáš). The minimum value
for the pitch is now 1 instead of 0.
* Improved the documentation with better examples, spelling error fixes, anchor links as well as some general cleanup and
restructuring (thanks Damien).
* Fixed a bug where the engine would throw a fatal error and then lock up if an include file was not found (thanks Damien).
* Modified the installer settings slightly, making it necessary to uninstall the previous version before installing this one.

Alpha version 0.4:
* Rewrote the string_split function to work with the new script interpreter, so it is now part of the engine again.
* Added a function called number_to_words, with which you can translate a numeric value into English text. This is useful for
speaking scores etc.
* Added a class to speak numbers intelligently using concatenated sound files to the include library.
* Fixed a bug that would cause the program to exit if the pitch was set to a small decimal value between 0 and 1 (thanks Oriol).
* Fixed a bug where the pan would be retrieved incorrectly by the engine (thanks Oriol).
* Changed the internal window handling so that it is now done asynchronously in the same thread as the main script execution,
preventing a few potential multithreading errors and speeding up some window related function calls.
* Compilation errors and warnings are now displayed in a multiline edit field, rather than in multiple message boxes with no easy
way of copying the information or getting a general overview.
* Fixed a serious memory leak where the audio resources would be left in a total mess at script termination.
* The tutorial has been updated with some new information in the section about arrays (thanks Damien).
* The end user license agreement has been added to this help file.

Alpha version 0.3:
* Fixed a few sneaky bugs that would cause BGT to crash when using some internal engine object properties in while conditions
and in arrays (thanks Damien).
* Fixed a bug where the command line could sometimes be filled with junk in the end (thanks Damien).
* The ranges of the properties in the sound object have been radically changed, and a few new ones have been added such as
pitch and sample_rate.
* Rudimentary error checking has been implemented (see the get_last_error function).
* The tutorial has been updated with a new section about switch ...case statements, and a few explanations have been added on
if statements when comparing booleans versus other values etc.
* A bunch of spelling mistakes have been fixed in the language tutorial (thanks Lukáš).
* The installer has been changed slightly to allow users who don't want a start menu folder to disable it, and the BGT end user
license agreement has been added also.
* An include directory has been added to the distribution where you will currently find a menu class.

Alpha version 0.2:
A new script interpreter has been integrated with the engine, there are so many major changes so that it is impossible to list them
here. In short, however, nothing at all is compatible with the 0.1 release. The language tutorial has undergone some major
modifications, there are some new methods in the sound object and a new object called dictionary, plus a whole lot more.

Alpha version 0.1:
First public alpha release.

