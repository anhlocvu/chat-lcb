Introduction to the Nonvisual Gaming Toolkit:
	Thank you for downloading the Nonvisual Gaming Toolkit (NVGT) by Sam Tupy Productions and contributors. We hope you find the time that you invest in this engine to be worthwhile!

	The NVGT documentation contains:
	* The NVGT user manual; containing everything from programming resources to general tutorials to example games,
	* API references; Class/function references that specify exactly how to use each feature of the engine,
	* Advanced topics; Documentation on the c++ side of the NVGT engine including how to build the engine from source, modify it's security functions, write documentation for the engine and more,
	* Appendix; data tables that don't fit in the API references, miscellaneous topics, attributions and license agreements.

Documentation status:
	It should be noted that this engine was only released to the public on May 31st 2024, and that the documentation is not yet complete as a result. Though the engine had been in development for years previously, the decision to open source it, and thus to write documentation for it, only took place a couple of months before the initial release. The choice was made to release the engine in an unpolished state specifically so that the community can help improve it, otherwise the engine would have remained private for years while only a couple of people worked in their spair time to complete the documentation. The engine is very usable and a growing community exists on Discord for any questions you might have, certainly we welcome all users! However those involved with the engine's development are also not personally advertising the engine yet due to it's unpolished state. Contributions to the documentation and other materials are therefore very appreciated, though you may want to hold off if you are the type looking for a particularly polished and pretty looking product at this time. Whether you decide to try the engine now or wait until the documentation is complete, we hope you enjoy your time engaging with NVGT and thank you for trying it!

offline documentation:
	The latest version of this documentation can be downloaded for offline viewing in several formats using the following links.
	* [nvgt.chm (compressed html help file)](https://nvgt.dev/docs/nvgt.chm)
	* [nvgt-html.zip (rendered html)](https://nvgt.dev/docs/nvgt-html.zip)
	* [nvgt-markdown.zip (github flavored markdown)](https://nvgt.dev/docs/nvgt-markdown.zip)
	* [nvgt.txt (plain text)](https://nvgt.dev/docs/nvgt.txt)


NVGT User Manual:
	Getting Started:
		Welcome to the Nonvisual Gaming Toolkit! This user manual contains all of the information you will need to get up and running using this powerful engine! whether you are new at programming, someone who used to create games using the Blastbay Game Toolkit or an advanced programmer who just wants an easy, no-hassle method of creating games, you're sure to find what you need in this manual to not only get up and running, but to also develop some confidence in it's usage.


	Upgrading From BGT:
		Since for a while this is where most people who have heard of NVGT will likely start, we will begin by describing the relationship between NVGT and BGT, and more importantly the steps required to upgrade a BGT project to an NVGT one. If you have not heard of BGT or are completely new to game development and are wondering how NVGT can help you, you can probably skip this and move to the fresh start topic.

		BGT stands for the Blastbay Gaming Toolkit, and is what initially inspired NVGT to be created after BGT was made into abandonware.

		It is important to note that NVGT is not officially endorsed by BGT's developers and contributors, and NVGT is not in any way derived from BGT's source code. The only shared assets between NVGT and BGT are some of BGT's include files such as the audio_form and sound_pool used by many bgt games, which are released under a zlib license.

		our goal is to make the transition as seamless as possible from BGT to NVGT, but here are some things you should know when porting an existing BGT game.

	Missing features:
		NVGT currently has some missing features compared with BGT. These may be added in the future, or have already been replaced with more flexible options.
		* Security functions such as `file_encrypt`, `file_decrypt`, `file_hash` do not exist in NVGT. These may or may not be added, but in the meantime you can encrypt/decrypt/hash files manually by using string_aes_encrypt/decrypt/hash_sha256/sha512.

	Differences:
	Functional:
		Some functions which have the same name as those in BGT work slightly differently.
		* `get_last_error`: Currently doesn't trigger in all the situations that BGT does.
		* When refering to an array's length, use length as a method call and not as a property. For example, you would use `array.length();` rather than `array.length;`.
		* Screen reader functions no longer need to specify the screen reader. If an active screen reader is detected it will automatically speak through there.
		* Data encrypted with BGT cannot be decrypted with NVGT, and vice versa.
		* The `sound::stream()` method does exist in NVGT, but it's simply an alias to `sound::load()`. For this reason it is recommended that you change all your `stream()` calls to `load()` instead. The `load` function performs an  efficient combination of streaming and preloading by default.
		* `load()` contains a second optional argument to pass a pack file. `set_sound_storage()` is no longer used for the moment. However, you can set `sound_pool.pack_file` or even the global `sound_default_pack` property to a pack handle to get a similar effect.
		* Take care to check any method calls using the `tts_voice` object as a few methods such as `set_voice` have changed from their BGT counterparts.
		* When splitting a string, matching against \r\n is advised as BGT handles this differently. This will result in not having spurious line breaks at the ends of split text.
		* The `settings` object no longer writes to the registry, but instead has been replaced by the settings.nvgt include which wraps the previous settings object API, but instead writes to configuration files in various formats.
		* The dynamic_menu.bgt include is now called bgt_dynamic_menu.nvgt. This is because NVGT now includes it's own menu class called menu.nvgt.
		* There is a type called `var` in the engine now, so you may need to be careful if your project contains any variables named var.
		* It's worth noting that unlike BGT, NVGT by default attempts to fully package your game for you including sounds, libraries, documents and any other assets into a .zip file or similar on other platforms intended for distrobution. If you don't like this behavior, you can create a file next to nvgt.exe called config.properties and add the line build.windows_bundle = 0 which will cause NVGT to just produce a standalone executable like BGT did, though you now may need to copy some libraries from the lib folder for the compiled product to run. For more information, see the Compiling your Project for Distribution topic.
		* In bgt, you could include pack files with the `#include` directive. In nvgt, we've decided that this should only include code. To embed packs, you can instead add the line `#pragma embed packname.dat` to your project, the extension can be any of your choosing for both code and packs this way.
		* Unlike BGT, you can embed multiple packs into an exe. For BGT compatibility, therefore, opening a pack file called `*` will open the first available pack, but it is recommended that you specify a filename such as `*mypack.dat`.
		* The `generate_profile` function returns a string which you can write into a file if desired.
		* Encrypting/decrypting sounds is done with the `asset_encryptor`/`asset_decryptor` datastreams, rather than with `file_encrypt`/`file_decrypt` functions. You can play encrypted sounds by setting the `sound_default_decryption_key` property.

	Nominal:
		The names of many functions have changed in NVGT. Some have even been moved into the objects they are linked to.

		Below is a list starting with the BGT function and the NVGT equivalent:
		* clipboard_copy_text: clipboard_set_text
		* clipboard_read_text: clipboard_get_text
		* generate_computer_id: generate_system_fingerprint
		* is_game_window_active: is_window_active
		* show_game_window: show_window
		* screen_reader_is_running: screen_reader_detect (see function in API reference for differing usage)
		* screen_reader_speak_interrupt: screen_reader_speak
		* screen_reader_stop_speech: screen_reader_silence
		* absolute: abs
		* arc_cosine: acos
		* arc_sine: asin
		* arc_tangent: atan
		* ceiling: ceil
		* cosine: cos
		* power: pow
		* sine: sin
		* square_root: sqrt
		* tangent: tan
		* serialize: dictionary.serialize
		* string_compress: string_deflate
		* string_contains: string.find
		* string_decompress: string_inflate
		* string_decrypt: string_aes_decrypt
		* string_encrypt: string_aes_encrypt
		* string_hash: separated into two functions (string_hash_sha256, string_hash_sha512)
		* string_is_alphanumeric: string.is_alphanumeric
		* string_is_digits: string.is_digits
		* string_is_lower_case: string.is_lower
		* string_is_upper_case: string.is_upper
		* string_len: string.length
		* string_replace: string.replace
		* string_reverse: string.reverse_bytes
		* string_split: string.split
		* string_to_lower_case: string.lower
		* string_to_number: parse_float/parse_double
		* string_to_upper_case: string.upper
		* The five substring functions string_left, string_mid, string_right, string_trim_left, string_trim_right, have been replaced with just two methods: string.substr and string.slice.

		The following key constants are also different:
		* KEY_PRIOR: KEY_PAGEUP
		* KEY_NEXT: KEY_PAGEDOWN
		* KEY_LCONTROL: KEY_LCTRL
		* KEY_RCONTROL: KEY_RCTRL
		* KEY_LWIN: KEY_LGUI
		* KEY_RWIN: KEY_RGUI
		* KEY_LMENU: KEY_LALT
		* KEY_RMENU: KEY_RALT
		* KEY_LBRACKET: KEY_LEFTBRACKET
		* KEY_RBRACKET: KEY_RIGHTBRACKET
		* KEY_NUMPADENTER: KEY_NUMPAD_ENTER
		* KEY_DASH: KEY_MINUS

	BGT compatibility helper:
		You can `#include "bgt_compat.nvgt"`, which implements a lot of aliases to BGT-compatible functions and classes. However when doing this, please be aware of the following limitations:
		* Using built-in functions may improve performance so this is more or less a stop-gap to get you up and running quickly; however if you wish for your code to just run, bgt_compat will certainly be of use.
		* Please note you will not be able to use this script if you have disabled global variables.
		* Since the screen reader internal functions no longer require you to specify a screen reader, the reader parameter is ignored.
		* The hardware_only parameter of generate_computer_id is ignored.


	Fresh start:
		This is the starting point for anyone who is interested in game development and who has found this engine, but who does not have much experience with programming concepts or the history of older game engines that are similar to this one.



	Effective Programming and Game Development with NVGT:
		Please note that this tutorial is still incomplete and though you should read it as a beginner to gain new information, you should be ware that it will cut off abruptly as the author has not come forward to complete what they started at this time.

	What is Game Development?:
		If you are an experienced programmer looking to get skilled up with NVGT as fast as possible, you can safely skip this section.

		To someone who has never programmed before, the development of games might seem like an insurmountable task. This manual hopes to help with the steep learning curve of programming, and to get you ready to make the games of your dreams all on your own.

		There are many things that need to be considered carefully in the development of a game. One of the most curious is programming; very quickly, programming stops aligning with real-world concepts. The precise instructions and rigid syntactic rules are often the bane of a new programmer's existence.

		As you learn, you will begin to "speak" programming more intuitively. You will eventually realize that programming languages are nothing more than a readable abstraction over common paradigms: variables, functions, conditions, loops, data structures, and many more. Start thinking of programming as a new way to think and construct new ideas, rather than a way of expressing already-imagined thoughts and ideas.

		And if you don't understand these paradigms yet, don't worry - reading this manual, my hope is that you will learn to make games with nvgt, but I have a secondary goal: by the end of this text, I hope that you will also be confident with programming in general, as well!

		Especially in the world of audiogames, whose developers are often small teams at most, skill in game design is correlated with skill in programming; this manual will attempt to teach you both. However, game designing itself can also be made into a career!

		When making games, you must first consider what your game will be: what are the rules of the game? What is the lore, if it has a story? Plan out your project in detail, so that it's easier to turn it into code later on. This is especially true of projects that have a plot, or at least backstory.

		As well as coding and designing the game, sounds must also be found or created; many high-quality resources can be found for this, be they free or paid.

		It is also the case that a released game is not necessarily a finished one: multiplayer games need administration to keep them running, and games might need to be maintained or updated to fix bugs that you did not encounter until post-launch.
	Your First NVGT Script:
		A script can also be considered one "unit" of code. You have probably seen many of these before: .py, .js, .vbs, perhaps .bgt, and now .nvgt. A script can be executed, meaning that the code contained within will be run by nvgt.

		Usually, one game will consist of not just one, but many scripts, each of which has some specific function or purpose. For now, though, a single script will be sufficient for our needs.

		A common tradition among programmers is to write a program that outputs "hello, world!" to the screen, to make sure whichever language they're using is working properly.

		Let's do that now and say hello to NVGT!
		Open a file in your text editor of choice (I recommend notepad++) and paste the following code inside:
		void main(){
		    alert("hello", "Hello, world!");
		}
		Now, save this file as hello.nvgt, and head over to it in your file explorer.

		Press enter on it, and you should see a dialog box appear, with our message inside!

		Congratulations - you've just written your first nvgt script!

		This script might look simple, but there's actually quite a bit going on here. Let's analyze our various lines:

		void main(){ - this is the beginning of a function (more on those later) which doesn't return anything, and takes no arguments, or parameters.

		The { afterwards opens a code block. It's customary in many codebases to put the { on the line which opens the block, and to put the closing brace on a new blank line after the block ends.

		Inside our block, we have one line: alert("hello", "Hello, world!");

		Let's break it down:

		alert() is a function in nvgt that opens a dialog box containing a message. It supports many arguments, but most are optional. Here, we have used the two that are not: title, and text.

		We've separated our parameters by a comma and a space, similar to what we do when listing items in English, although we don't need to use "and" like we do in English, and there is only one rule to remember: a comma after every item, except the last.

		Our parameters are enclosed in parentheses () but why? This tells NVGT what we'd like to do with the alert function: we would like to call it (calling meaning running the code inside based on some values). Finally, we end the line with a semicolon ; that tells NVGT this piece of code is finished.

		Together, alert("hello", "Hello, world!"); is known as one statement, the smallest unit of code that can be executed.
		As you can see, functions aren't so daunting after all!

		However, there's one missing piece of this puzzle: what's the deal with the main() function?

		Actually, it's arbitrary, although extremely common. Many programming languages (rust, c, c++, java, and NVGT, to name a few) require you to use the main() function as what's called the "entry point": in other words, NVGT will call the main function, just as we called the alert function. If it doesn't find the main function, it won't know where to start, and will simply give you an error stating as much.

		It's very possible to write scripts without a main function. These are sometimes called modules, and sometimes called include scripts, helper scripts, or any number of other names. In NVGT, since a module is quite a different (and advanced) concept, we call them include scripts.

		There's something interesting which nvgt can do with this script: not only can you run it, but you can also pack it into an exe file, ready to be shared with whoever you wish to have it. The advantage is that the exe will obfuscate your code and protect it from bad actors, which is especially useful for multiplayer projects!

		Not just that, but anyone can run your game if you compile it, whether or not they have nvgt installed on their computers.

		It's easy to do: when you've selected your script in windows explorer, don't run it. Instead, press the applications key (alternatively shift+f10 if you don't have one of those) and a "compile script (release)" option should appear.

		Click it, and a new file should appear next to the script: hello.exe.

		Running this file, you should get a similar effect to just running the script, but the code is now protected and can no longer be accessed.

		Now that we've learned the basics of nvgt scripts, let's move on and create a simple program!
	Learning Project: Calculator:

		Let's make a calculator program with nvgt. We should be able to type mathematical expressions into it, and then get results out. We'll then build on our program by adding more features, and hopefully learn lots along the way!

		You'll encounter many new programming concepts at once, so don't feel discouraged if you are lost. All will be explained!

		In this chapter, we'll learn about many of the fundamental concepts of nvgt, and of programming as well! I will try to make this interesting, but learning the basics can admittedly be boring. It is not shameful if you need a coffee or two reading through this.

		And once we've learned the basics, we'll create several versions of our project, each one better than the last!

		I will first explain the nvgt and programming concepts we'll need to understand it, using some short examples. They are called snippets, and won't all actually run. They are just to demonstrate.
	comments:
		Sometimes, you might want to document what a particularly complex bit of code does, or perhaps remind yourself to make it better later.

		For these reasons and more, programming languages usually have some way of keeping some notes in your code that they will not attempt to execute.

		These are usually called comments, and nvgt has three types of them:

		The single-line comment is written with a //, and looks like this:
		// Hello!
		The multi-line comment is written slightly differently, enclosed by a /* and a */. It looks like this.
		/*
		Hello!
		This is a multiline comment
		Lorem ipsum
		Dolor sit amet
		*/
		Lastly, the end-of-line comment is sort of like a single-line comment, but goes after a line of code. For example:
		int drink_price=5; // should this be able to store fractional values?

		As well as documenting your code, comments (especially multi-line ones) can be used to surround code which temporarily doesn't need to be run, or is broken for some reason.

		This is called "commenting out" code. Code which is commented out will be ignored by the compiler, just like regular textual comments.

		Example:
		void main(){
		//alert("hello", "hello, world!")
		alert("how are you", "My name is NVGT!");
		}
		If you run this, you will only see the "how are you" dialog box, and not our "hello" one. You see, I made a little error when I was writing that line - can you spot why we might have commented it out?

	include scripts:
		The truth about programming, as with anything, is that organization is the most important thing. Always remember this: reading code is more difficult than writing it.

		As a programmer, your highest-priority task is to get the code working. But arguably just as important is to keep it working.

		Imagine you are a version of yourself, with similar programming experience. However, you have never read your project's code before. If you can intuitively understand what all of its components do, then you have succeeded in writing readable code!

		Another thing experienced programmers like to have is modularity. This philosophy dictates that as little of your code as possible should depend on other code.

		I'll explain more about this later, but an include script is how one achieves modularity in nvgt: by using the #include directive at the top of your program, you can load in another file of code, adding its code to your own in doing so.

		NVGT ships with a host of include scripts (or includes for short), which you are free to use to speed up the process of game development.

		For example, the speech.nvgt include has functions for making your game speak out text, without needing to worry about what screen reader or sapi voice the user has.

		If you wanted to use the speech.nvgt include, you would put it at the very top of your script, like this:
	include "speech.nvgt":

		Why the #? In nvgt, # signifies what's called a preprocessor directive: usually one line of code, these are used before your program is run to change something about it.

		NVGT has what's called an include path. It searches multiple folders for your include scripts, and if it can't find them, it will give you an error. It works a bit like this:
		1. Search the directory of the script from which another script was included
		2. Search the include folder in nvgt, in which all the built-in includes are stored.
		[add more later here maybe]

		Here is a full code example, which you can copy into an nvgt script and run.
	include "speech.nvgt":
		void main(){
		    speak("Hello from the NVGT speech include!");
		}

	variables:
		If you know about variables in algebra, then you will have a similar concept in your head to those in programming. They are, at their core, names of data.

		Quite a few things in nvgt are variables: functions, normal variables, classes and many more.

		In this section we will learn about how to make some simple variables, and how to change them.
	integers (ints) and unsigned integers (uints):
		In NVGT, there are two main types of numbers, integers and floating-point numbers.

		The easiest to understand is an integer, otherwise known as a discrete number. We'll learn about those first, then move on to floats, the type we will make the most use of in our calculator project.

		Here are some examples of declaring the two kinds of integers:
		int x = 3;
		x = -3;
		uint y = 3;
		y=-3; // Oh no!
		As you can see, we used both kinds of integers. One is called an int, as we'd expect, but the other is called a uint. What does the u mean? You might have already guessed!

		We'll talk about that in a second. And if you don't want to learn about binary now, it's enough to know that unsigned ints sacrifice the ability to store negative values for double+1 the positive number range.

		First, let's break down our int declaration statement

		`int` is the type of variable (int)

		`x` and `y` are the "identifier" of a variable. In other words, its name.

		`=` is the assignment operator. The first time you assign a value to a variable, it is called initialization.

		After the =, a value is assigned to the variable. Then, the ; is used to end our statement, making it complete and ending the line.

		You'll notice that only the first reference of a variable needs its type; this is because this is the declaration of the variable, whereas the second line is a reassignment of the same variable and does not need to be re-declared.

		You can also declare what are called global variables. I'll give a full example to demonstrate this.
		int unread_emails = 20;
		void main(){
		    alert("important", "You have " +unread_emails + " unread emails!");
		}
		As you can see, despite the fact that the global variable was not declared within the function (or its scope), we can still use it. This is because it's not declared in any function, and can thus be used from all functions in your program.

		This author personally does not recommend much usage of globals. A more elegant way to use variables in lots of different functions at once will be demonstrated shortly. The exception is consts, which will also be discussed later.

		To create the message, I used the string concatenation operator to glue one string, one variable, and another string together. This will be further explained in the section on Strings.

		As we know, inside a function (but not outside) variables can be changed, or re-assigned. You might have realized that changing a variable's value in relation to itself (like giving the player some money or points) is a handy side effect, since you can simply reference a variable when re-assigning it, like this:
		int level = 2;
		level = level + 1;
		But there is a simpler, more readable way of doing this, which saves lots of time (and cuts down on typos!). 

		If you want to change a variable in relation to itself like this, you use what's called compound assignment.

		This combines an operator, like an arithmetic operation or string concatenation, with the assignment operator.

		For example, we could rewrite our previous code using compound assignment:
		int level = 2;
		level +=1;
		As you can see, it's much cleaner!

		Here's a full example to consolidate what we've learned. You can copy and paste it into an nvgt script and run it. We'll also demonstrate includes and comments again!
	include "speech.nvgt":
		int g = 3; // a global variable
		void main(){
		    /*
		    This program demonstrates integers in NVGT, by declaring one (a) and performing a variety of arithmetic operations.
		    After each operation, the value of a will be spoken.
		    */
		    int a = 0; // This is the variable we'll use. 
		    speak("a is now " + a);
		    a+=2;
		    speak("After adding 2, a is now " + a);
		    a*=6;
		    speak("After multiplying by 6, a is now " + a);
		    a /=3;
		    speak("After dividing by 3, a is now " + a);
		    //something new!
		    a -= g;
		    speak("After subtracting g, a is now " + a);
		}

	bonus: binary 1100101 (or 101):
		To understand signed and unsigned integers (and to understand integers) we must first understand binary. Otherwise known as base 2, it is a system based on 0s and 1s. It's most well known for appearing on terminals in poorly written movies about hackers, but it is integral to understand it as a programmer, so let's talk about it.

		The unsigned integer version of binary is actually easier to explain, so we'll start with that one.

		Consider a row of bits. In base 2, we can already surmise that the maximum possible value is 2^bits-1. NVGT's ints are 32-bit, although it does also support int64 and uint64 types if you want them.

		The unsigned integer (uint) type in nvgt, thus, can store a maximum value of 4.294 billion. This is a huge number, and is suitable for most, if not all, requirements. The unsigned 64-bit integer type can store a value of up to 18.446 quintillion, which is more than two billion times the world population and more than a thousand times the amount of money in circulation in the entire economy, in US cents.

		The first bit on the left is worth 2 raised to the n-1th power, where n is the number of bits.

		If the bit is set to 0, it means no value is added to the total in base 10. If it's set to 1, you add its worth.

		From left to right, each bit is worth half the bit before it. Let's give examples with 8 bits, since that's much easier to think about than 32 bits.

		Consider this set of bits: 01100101

		The leftmost bit in this group would be worth 128, since that's the value of 2^(8-1).

		But it's set to 0, so we don't do anything

		Right another bit, and we get a bit of worth 64, set to 1. So, we add 64.

		Next, another 1 bit, with a value of 32, which we add, giving us 96.

		Next, two 0 bits, each worth 16 and 8. We will ignore them.

		Another 1 bit, worth 4. Let's add it, for a total of 100.

		Then, a 0 bit, worth 2, and another 1 bit, worth 1. Adding the last 1 bit, we have our final total, 101.

		This is all you need to know about unsigned binary representation.

	float variables:

		The main difference between ints and floats is that floats can store fractional values, whereas ints are restricted to exclusively whole numbers. Although they do come with some drawbacks, this makes floats more suitable for tasks where a high level of precision is needed. They are also useful for dealing with imprecise, but extremely large, numbers.

		There are two main types of floats in nvgt: float and double.

		Float is a 32-bit (or single-precision) variable, and double is a 64-bit variant which can store a greater number of decimal digits and higher exponents.

		In most cases, you should be okay to use a double, but this is not always required and is often not a good choice anyway.

		The inner workings of floats are beyond the scope of this tutorial, but it's enough to know that computers don't think about fractional values like we do: the concept of decimal does not exist to them.

		Instead, they use a binary representation, called the IEEE754 standard.

		You cannot rely on floats storing a number perfectly. Sometimes, the IEEE754 standard has no exact representation for a number, and its closest equivalent must be used instead.

		To demonstrate this, run this script. The result should be 1.21, but it isn't.
	include "speech.nvgt":
		void main(){
		    double result = 1.1 * 1.1;
		    screen_reader_speak(result, false); // implicit cast from double to string
		}
		As you  can see, the value is very close, but not quite right. We even used the double type, with 64 bits of precision, but it wasn't enough.

		There are several ways to get around this, but we don't need to worry about them for this project, so let's learn about another very useful type of variable: strings!

	string variables:
		The easiest and most reliable way to think about string variables is that they are text. "hello" is a string, "this is a test" is a string, and "1" is a string (this last one is confusing but will be explained shortly).

		We have actually seen string variables before. When we were making our hello world program, we used two of them for the title and text parameter to the alert function.

		Now knowing about variables and their identifiers, you can probably see why we used quotes (") around them, and why that is necessary.

		If we hadn't, "hello, world!" would've ended up being interpreted as two function parameters, the variable identifiers hello and world, neither of which existed in the program.

		NVGT would not have liked this at all; in fact, it would've thrown numerous errors our  way in response.

		So, quotes must enclose strings to let NVGT know that it should ignore the text inside - the computer doesn't need to know that it's text, only that it's data like text, which it can then show to the user for them to interpret.

		It's almost, if not quite,  like delivering letters: you don't know or care about the information in a letter (and if you did, then your manager probably needs to talk to you!) but the letter's recipient does.

		In the same way, NVGT will happily place text in quotes in strings, which can then be passed to functions, put into variables, or concatenated onto other variables or strings.

		In this case, it was assigning them to the title and text variables, arguments of the alert function.

		String variables are created using a similar syntax to the int variable we just saw:
		string name = "Rory";
		You can also create a string with multiple words:
		string message = "How are you today?";
		Or even a string with non-ascii characters:
		string message2 = "Hello, and 你好 👋";
		Just like integer variables, strings also have operations which can be performed on them. By far, the most common is concatenation.

		Concatenation is the process of stringing strings together. The + symbol is used for this, and compound assignment with += is also supported.

		Let's see how it works:
		string sentence = "The quick brown fox";
		sentence += " jumps over the lazy dog";
		To give you some familiarity with string concatenation, let's go over a full example. Copy this into an NVGT script, and run it:
	include "speech.nvgt":
		void main(){
		    int a = 1;
		    int b = 2;
		    string c = "1";
		    int d = 2;
		    string result1 = a + b;
		    string result2 = c + d;
		    speak("a + b is " + result1);
		    speak("c + d is " + result2);
		}
		The output should be:

		a + b is 3

		c + d is 12

		Is that what you expected?

		What's happening here is called casting, specifically implicit or automatic casting. In programming, casting means converting some value of one type (int) to another type (string).

		When we calculate result1, we perform addition on a + b (1 + 2) and get 3, which makes sense.

		But when we calculate result2, NVGT automatically converts d, an int, into a string, making it "2", so it can be concatenated.

		It then ignores the data inside, and just adds it together. So, instead of having 1 + 2, you get "1" + "2" - which is just two pieces of data combined together into one string, making "12".

		This is why strings are so useful: they can hold any data in a text-like form, and then the meaning of that data can be interpreted by something else. In this case, it is output to our screen readers, and we can interpret it; the sound waves by themselves do not hold any semantic information.

	boolean variables (bool):
		Leading up to a powerful idea called conditionals, boolean variables are another fundamental datatype in programming, and are usually present in some form in programming languages.

		Named in honour of the mathematician and logician George Boole, the variables can store only two possible values: true or false - or 1 or 0, on or off, yes or no.

		They are extremely powerful and there are quite a few things you can do with them, but most of them don't really make sense without conditionals. Still, we can give a basic example, using not (!), a logical operator:
		void main(){
		    bool state = true;
		    speak("State is: " + state);
		    state = !state;
		    speak("Flipped, state is: " + state);
		}
		This shows how to declare a bool: it's fairly similar to other variables. Unlike strings, the values true or false do not need to be put in quotes, despite the fact that they are not variables in the traditional sense. These variables are actually consts, which means you can never accidentally overwrite their values; trying will yield an error.

		That's all we'll learn about variables for now. We'll come back to them later on, but for our calculator project, this is all that we'll need to know.

	Const Keyword:
		For this project, the last thing we will explore regarding variables is consts.

		Const variables (or constants) are variables which can never change after they are first assigned. This must be done when they are initialized.

		They are particularly useful in avoiding "magic numbers": using numbers directly in our code is bad!

		Let's write some code to demonstrate:
	include "speech.nvgt":
		void main(){
		    speak("Welcome to the camping store! You need to have 30 dollars to buy a folding chair.");
		}

		This looks fine, but we're using this value in only one area of our code (mostly because there is only one place in which to use it, but that's beside the point).

		Suppose, now, that we use the value 30 in many areas: not just telling the user how much it is to buy a chair, but also for the logic of buying and selling them itself.

		This is also valid in that it works, but it is frowned upon.

		Consider this: inflation is making everything more expensive these days, so what if we need to raise this price to 35 dollars next year?

		The answer to that question is that it would be a coding nightmare! We would have to go through our code, painstakingly changing every reference of the value 30 to 35. But we could get it wrong: we might accidentally make one of the values 53, or change the number of centimeters in a foot from 30 to 35 in our frantic search - wouldn't it be much better if we only had to change the value once?

		This is where consts will save the day!

		Using them , we could rewrite our code like this:
	include "speech.nvgt":
		const int price_folding_chair = 30;
		void main(){
		    speak("Welcome to the camping store! You need to have " + price_folding_chair + " dollars to buy a folding chair.");
		}
		Much better! Now we can use this value wherever we want to, and all we have to do to change it is update the one declaration at the top of the file!

	conditionals and the if statement:
		The if statement is perhaps one of the most overwhelmingly useful pieces of code. Its ubiquity is almost unparalleled and a variation of it can be found in just about every programming language in use today.

		Say you want to run some code, but only if a specific thing is true - the if statement is what you use. Let's give a full example:
	include "speech.nvgt":
		void main(){
		    int dice = random(1, 6);
		    if(dice == 1)
		        speak("Wow, you got " + dice + "! That's super lucky!");
		    else if(dice < 4 )
		        speak("You got " + dice + " - that's still pretty lucky, but aim for a 1 next time!");
		    else
		        speak("Ah, better luck next time. You got a " + dice + ".");
		}
		This small dice game is very simple. Roll a dice, and see how low a number you can get.

		There are three options: you get a 1 (the luckiest roll), 2 or 3 (the 2nd luckiest), or 4-6 (which means you lose).

		We can express these three options using the "if", "else if", and optional "else" constructions. They work like this:

		if(conditional)

		statement/block

		else if(conditional)

		statement/block

		else if(conditional)

		statement/block

		else

		statement/block

		These are slightly different than normal statements, because they do not always have the ; where you would expect, at the end: if multiple lines are to be run (a block, surrounded by braces) then you use the ; on the last line of the block, before the }.

		But what is a conditional?
		A condition is any value that returns either true or false. This can be something which is already a bool variable, or the result of a comparison operator.
		There are six comparison operators in nvgt, each of which takes two values and returns true or false based on their relationship.
		| Operator | Purpose                                              | Opposite |
		|----------|------------------------------------------------------|----------|
		| ==       | Checks if two values are exactly equal               | !=       |
		| !=       | Checks if two values are not exactly equal           | ==       |
		| <=       | Checks if a value is less than or equal to another   | >        |
		| >=       | Checks if a value is greater than or equal to another| <        |
		| >        | Checks if a value is greater than another            | <=       |
		| <        | Checks if a value is less than another               | >=       |

		There are also four logical operators which work on bools, one of which we explored in the section on booleans. They are:
		* && (and) returns true if the bools on the left and right are both true, but not neither or just one
		* || (or) returns true if either or both of the bools on the left or right are true
		* ^^ (xor) returns true only if one of the left and right bools is true, but not neither or both
		* ! (not) returns the opposite of the bool on the right (it takes only one bool and nothing on the left)

		Using these comparison operators and logical operators is how one creates conditionals to use in if statements, though remember that bools themselves can already be used directly.

		This is all a lot of information at once, so here's a full example to demonstrate:
	include "speech.nvgt":
		void main(){
		    int your_level = 5;
		    int monster_level=10;
		    int your_xp=50;
		    bool alive=true;
		    if(alive)
		        speak("You are alive!");
		    if(monster_level<=your_level){
		        speak("You manage to kill the monster!");
		    }
		    else{
		        speak("The monster is higher level than you, so it kills you!");
		        alive=false;
		    }
		    if(!alive)
		        speak("You are no longer alive!");
		    if(your_level*10==your_xp)
		        speak("Your level is equal to a tenth of your experience points.");
		}   
		This example demonstrates an important distinction we touched upon earlier: if statements which use a block instead of a single statement need to have the block surrounded by braces.

		Where a block might be used, a single statement can usually be used as well; the exception is functions, which must always use a block, whether or not it is a single line.

	loops:
		While the if statement is used to create conditional checks for whether certain pieces of code will be run, loops are used to repeat sections of code more than once.

		Loops have many uses, including infinite loops (which are already a well-known concept), loops that run a known number of times, and loops that we use to run through certain data structures like arrays.

		For advanced programmers, NVGT does not support iterators; the traditional c-style for loop must be used to iterate through arrays, although this is substantially more convenient because one can query their length.

		NVGT has three main types  of loop, which we will discuss in this section: while loops, while-do loops, and for loops. We will also discuss the break and continue statements.

	While Loops:
		The most simple form of loop is the while loop. It is written almost exactly the same as the if statement, and runs exactly the same, except that it will check the condition and run the code over and over, checking before each run, until the condition is determined to be false.

		Here is an example which you can run:
	include "speech.nvgt":
		void main(){
		    int counter = 1;
		    while(counter <6){
		        speak(counter);
		        wait(1000);
		        counter+=1;
		    }
		}
		This should speak out the numbers 1 through 5, with a second's delay between each.

		Just like if statements (as well as other types of loops), a while loop can also be written with only one line of code inside. If so, it does not need to be surrounded by braces.

		The while loop is the standard way to write an infinite loop: a piece of code that will run forever, perhaps until broken out of with the break keyword.

		To make an infinite loop, we simply use
		true
		as the condition, since it will logically always return true, every time.

	Do-While Loops:
		In while loops, the condition is checked before the code is run. This means that, just like an if statement, there is always the possibility that the code may never run at all.

		On the other hand, in do-while loops, the condition is checked after the code has run. As a result, code will always run at least one time.

		It's often up to the programmer which type they think is best, and there is no real standard. Whichever is more convenient and maps best to the situation can be used, and neither has a performance advantage over the other.

		Let's rewrite our counter example using a do-while loop:
	include "speech.nvgt":
		void main(){
		    int counter = 1;
		    do{
		        speak(counter);
		        wait(1000);
		        counter+=1;
		    } while(counter < 6);
		}
		As you can see, since the condition is at the end, we need to check it based on the value after the counter as updated, instead of before.

		If we had used
		while(counter < 5);
		as we had done with our while loop, the code would have only counted to 4, since the counter gets updated after it speaks its current value.

	For Loops:
		One of the most-used types of loop is something similar to our counter example from earlier in this chapter.

		There are a lot of lines here which for loops can compress into just one line; they make code easier to write as well as read later on.

		For loops also have an additional unique property, which is extremely useful, but will be discussed in a moment after some background.

		However, they are  admittedly difficult to grasp at first, because they're very different than both the while and do-while loops.

		Consisting of four parts, a for loop might look like this:
		for(declarations; condition; final)
		statement/block

		How it works:

		1. The declarations code is run, and a variable (or more) is declared.
		2. The loop starts to run.
		3. At the beginning of each loop iteration, the condition is checked. If false, the loop stops, potentially even before it has run once.
		4. Assuming the condition is true, The code inside the for loop is run.
		5. After that code has finished, the final step is run.
		6. Repeat steps 3-5

		Note: all three parts of a for-loop are optional. For instance, you may omit the declaration by simply writing nothing before the first ;

		As you can imagine, a for loop would help us rewrite our counter example in a much more readable way.

		Let's go ahead and do this now:
		void main(){
		    for(uint i = 1; i < 6; i ++){
		        screen_reader_speak(i, false);
		        wait(1000);
		    }
		}

		This example will yield the same results as the previous one, but it does it in a way which is more concise. Pretty code is very important for your sanity!

	Break And Continue Statements:
		There are times when you might want to get out of loops completely. This is called "breaking out", and it's very easy to do:

		break;
		There are no additional components to this statement, as there are in some other languages such as rust.

		This is particularly useful in infinite loops (typically created by simply using true as the condition in a while loop)

		When this is done, the loop immediately ends, and will iterate no longer.

		If you want to break out of a loop, but still want it to try and run again (IE. not run the rest of this iteration), the continue statement can help.

		As is the break statement, the continue statement is simply written on a line by itself:
		continue;
		This immediately ends the current iteration. For while and do-while loops, nothing more happens, and the final component in a for loop is run. Then, the next iteration may begin.

		The reason that for loops behave differently is by design. It is not possible to achieve this behaviour in any other way.

	functions:
		Functions in nvgt are pieces of code that can be "called" or executed by other parts of the code. They take parameters (or arguments) as input, and output a value, called the "return value".

		The return value is so named because it gives some critical piece of information back to the calling function, without which work cannot continue.

		Let's use baking a cake as an example: in the process of baking a cake (simplifying greatly) you put the batter in the oven. The oven cooks the cake, and then you open it up and take a cooked cake out.

		You can't do something else related to baking a cake while the oven is baking, because you don't have the cake: the oven does. In the same vein, only one function can be running at once, and we say that function is currently executing.

		When a function ends, execution returns to the function from which it was called, just like taking the cake back out of the oven. It returns, along with execution, whatever is specified using a return statement (discussed shortly).

		Here is a snippet for the purpose of example:
		int add(int a, int b){
		    return a + b;
		}
		Frankly, this code is a needless abstraction over an already-simple task (the addition operator) and you should almost never do it like this in production. Nonetheless, it's a good example of what a function can do. It takes data in, and then outputs some other data.

		The way we declare functions is a little bit strange, and this example is packed with new ideas, so let's break it down piece by piece:

		int add(

		The beginning of the function's declaration, letting the compiler know that it is a function with the return datatype (int), the name (add) and the left parenthesis

		int

		the type of the first variable

		a, 

		the name (identifier) of the first parameter/argument variable, and a comma and space to separate it from the next variable, just as we use when calling functions

		int b)

		The declaration of the second parameter/argument variable, another integer, and a right parenthesis to tell the compiler there are no more parameter variables

		{

		The beginning of our function (remember, there must always be braces enclosing functions, even if the execution step is only one line in length.)

		Then, the next line:

		    return a + b;

		This is the only line in our function, and returns an expression evaluating to an int variable. It adds the a and b variables' values together.

		}

		The end of our function, which lets the compiler know that we're back in the outer scope (probably global)

	Bonus: Some notes About References (Advanced):

		If you are new, you can skip this brief section for now, as it's discussed in another article in great depth for easier understanding.

		There are a couple of  common misconceptions and mistakes made by even experienced coders when it comes to function parameters.

		For performance, it may seem intuitive to declare your primitive functions as const x &in references, but this is almost always useless, except in the case of strings.

		The reason for this is the fundamental property of a reference: a pointer itself is a value of 8 bytes storing a memory address. As opposed to most primitives (ints etc), there is no advantage, as the new bytes still need to be allocated - it is just a different value that is placed into them (the address instead of a copy of the value).

	Arrays (Lists):
		If you have programmed prior to reading this manual, you may have seen them before: an array is angelscript's equivalent to a vector or a list.

		Via this powerful data structure, we can store lots of the same type of variable in one neat package.

		In NVGT, arrays are dynamic. This means that you can add and remove from them at any time you'd like, as opposed to arrays in c or rust, which are trickier to expand or remove from.

		Before we move on, here is a quick example:
		void main(){
		    int[] scores = {20, 30};
		    scores.insert_last(50);
		    screen_reader_speak("The scores of the players in this game are:", false);
		    for(uint i = 0; i < scores.length(); i ++)
		        screen_reader_speak(scores[i], false);
		}

		The variables in an array are called "items" or "elements", and you can imagine them as being stored in a line (ordered).

		As a consequence, we can access the 1st, 2nd, or any other item in our line of elements, via "indexing".

		Slightly more complicated to think about, however, is that arrays in NVGT use "0-based indexing".

		This is true of most programming languages, with a notable exception being lua.

		0-based indexing means that, instead of the 1st item being at index 1, it is at index 0. Item 2 would be at index 1, item 3 at 2, and item 20 at 19.

	To be Continued:
		Unfortunately, this game development tutorial is not yet complete, you should stop reading at this point if you are a beginner.
		* classes (methods and properties)
		* datastreams and files


	Toolkit Configuration:
		One highly useful aspect of NVGT is it's ever growing list of ways that an end user can configure the engine. Whether trying to add an extra include directory, tweak an Angelscript configuration property, choose how warnings and errors are printed, deciding whether to print output on successful compilation or any number of other changes, NVGT provides various options and directives that should help you to get it functioning much closer to the way you desire.

		From `#pragma` directives to command line argument syntax to configuration files, this document attempts to keep track of all such available configuration options NVGT has to offer.

	Command line arguments:
		NVGT's compiler program has a variety of command line arguments you can pass to it which can alter it's behavior. Some of these options can also be set in configuration files, however an explicit command line option is very likely to override any contradictory options that are in a configuration file.

		Generally, these command line options are self-documenting. If you open a command prompt or terminal and just run the nvgt application directly with no arguments, you will be informed that you should either provide an input file or run `nvgt --help` for usage instructions. You can also actually just run nvgt -h

		In almost all cases, command line arguments have 2 methods of invocation, both a short form and a long form. The short form of a command usually consists of just a letter or two and is easier to type, while the long form of a command is always more descriptive and thus might be suited to an automation script where you want anyone reading such a script to be able to understand exactly what they are asking NVGT to do. What form you wish to use is completely up to you. In either case you should put a space between arguments, and mixing short and long form arguments is completely fine.

		The short form of an argument always begins with a single hyphen character (-) followed by one or 2 letters. For example, `-C` would tell NVGT to compile your script with debug information. Some arguments take values, in which case the value shall follow directly after the argument when using short form invocations. For example you could set the platform to compile with using the `-pwindows` argument.

		On the other hand, the long form of an argument begins with two hyphens followed by what usually amounts to a few words separated by hyphens. Usually it's just one or 2 words, but could be more in rare cases. For example, `--compile-debug` would tell NVGT to compile the given script with debug information. If an option takes an argument, you use an equals (=) character to define such a value. For example `--platform=windows` would tell NVGT to compile a script for the windows platform.

		Finally, a special argument, two hyphens without any following text, indicates that any further arguments should be passed onto the nvgt script that is about to run.

	Argument list:
		The following is a list of all available command line arguments, though note that it is best to directly run `nvgt --help` yourself encase this list is in any way out of date as nvgt's --help argument will always be more accurate because the text it prints is dynamically generated.
		* -c, --compile: compile script in release mode
		* -C, --compile-debug: compile script in debug mode
		* -pplatform, --platform=platform: select target platform to compile for (auto|android|windows|linux|mac|ios)
		* -q, --quiet: do not output anything upon successful compilation
		* -Q, --QUIET: do not output anything (work in progress), error status must be determined by process exit code (intended for automation)
		* -d, --debug: run with the Angelscript debugger
		* -wlevel, --warnings=level: select how script warnings should be handled (0 ignore (default), 1 print, 2 treat as error)
		* -iscript, --include=script: include an additional script similar to the #include directive
		* -Idirectory, --include-directory=directory: add an additional directory to the search path for included scripts
		* -V, --version: print version information and exit
		* -h, --help: display available command line options

	Configuration files:
		Both because command line arguments can become exhausting to type for every script execution, and because NVGT contains far too many configuration options to make command line arguments out of, NVGT also supports configuration files which can be used either on a global or a project level to further direct NVGT.

		Configuration files can either be in ini, json or properties formats and are loaded from multiple locations. Values in these configuration files are usually separated into sections in some form. Settings controlling the user interface typically go in the application subsection, while directives that provide control over Angelscript's engine properties are in a subsection called scripting. The way these directives in various subsections are defined depends on the configuration format you choose, however you can use any combination of configuration formats at once to suit your needs. Though all supported configuration formats are more or less standard, you can find examples of and notes on each format below.

		Configuration files at any given location have a priority attached to them to resolve directive conflicts. For instance, if a global configuration file and a project configuration file both contain the same option, the project file takes priority.

	Loaded configuration files:
		The following configuration files are loaded listed in order of priority from highest to lowest:
		* Script-dependent project configuration: script_basename.ini, script_basename.json or script_basename.properties in the same directory as the nvgt script about to be run, where script_basename is the name of the nvgt script without an extension (mygame.nvgt = mygame.properties)
		* Script independent project configuration: .nvgtrc (ini format) in the same directory containing the nvgt script that is getting executed or in any parent
		* Compiler dependent global configuration: exe_basename.ini, exe_basename.json and exe_basename.properties in NVGT's installed directory or any parent, where exe_basename is the name of nvgt's running executable without the extension (for example nvgt.json or nvgtw.ini)
		* Compiler independent global configuration: config.ini, config.json and config.properties in NVGT's installed directory.

	Supported configuration formats:
		NVGT supports 3 widely standard configuration formats, all of which have their own advantages and disadvantages. It is perfectly acceptable to create multiple configuration files with the same name but with different extension, all keys from all files will load and be combined. If the files nvgt.json and nvgt.ini both exist and both set the same key to a different value, however, the result is a bit less defined as to what key is actually used and typically depends on what file was internally loaded first by the engine.

		It is entirely up to you what formats you want to use in your projects.

	json:
		{"application": {
			"quiet": true
		}, "scripting": {
			"compiler_warnings": 1,
			"allow_multiline_strings": true
		}}
		Likely the most widely used and most modern, NVGT can load configuration options from standard JSON formatted input.

		This format receives the most points for standardization. Included directly within many higher level programming languages as preinstalled packages or modules, this is the easiest format to use if you need some code to generate or modify some configuration options used in your program. For example, a python script that prepares your game for distribution might write a gamename.json file next to your gamename.nvgt script that modifies some Angelscript engine options or changes debug settings before release, and sets them to other values during development.

		Furthermore if NVGT ever includes an interface to set some of these options in a manner where they save, .json files would be the format written out by the engine.

		The disadvantage to JSON though is that even though it is the most modern and widely recognised of nvgt's options, it's relatively the least human friendly of the formats. For example if a configuration option takes a path on the filesystem and one wants to manually set this option in NVGT, they would need to escape either the slash or backslash characters in the path if they go with the JSON option. Furthermore NVGT contains many boolean options that are just enabled by being present, where an option key has no needed value. While JSON of course supports setting a value to an empty string, it is certainly extra characters to type in order to just quickly tweak an NVGT option.

	ini:
		[application]
		quiet

		[scripting]
			compiler_warnings = 1
			allow_multiline_strings
		NVGT can also load configuration data from good old ini formatted text files.

		This format is probably the least standardized of the ones NVGT supports, likely due to it's origin from closed source windows. While on the surface it is a very simple format, there are a few deviations that exist in various parsers as the syntax is not as set in stone as other formats. For example whether subsections can be nested, whether strings should be quoted, whether escaping characters is supported or whether a value is required with a key name are all up to an individual ini parser rather than a mandated standard. In this case, the ini parser that handles these configuration files does not require that a value be provided with a key name for simple boolean options, escaping characters is not supported and strings do not need to be quoted.

		The biggest advantage to this format as it pertains to NVGT is the simplicity of setting several options in the same section without constantly redeclaring the sections name. One can just declare that they are setting options in the scripting section and can then begin setting various engine properties without typing the name "scripting" over and over again for each option.

		The biggest disadvantage to this format is the inability to escape characters. For example, typing \n in a configuration string would insert the text verbatim instead of a newline character as might be expected.

	properties:
		application.quiet
		scripting.compiler_warnings=1
		scripting.allow_multiline_strings
		Finally, NVGT is able to load configuration options from java style .properties files.

		The biggest advantage to this format is it's easy simplicity. The format just consists of a linear list of key=value pairs, though the =value is not required for simple boolean options that just need to be present to exist. Unlike ini, it also supports character escaping, causing \n to turn into a line feed character.

		The only real disadvantage to this format over ini is the need to keep constantly redeclaring the parent section names whenever defining keys, for example you need to type scripting.this, scripting.that where as with ini and JSON you can specify that you are working within a section called scripting before adding keys called this and that.

	Available configuration options:
		Many options listed here do not require a value, simply defining them is enough to make them have an effect. Cases where a value is required are clearly noted.

		The available options have been split into sections for easier browsing. While the method for defining options changes somewhat based on configuration format, a line in a .properties file that contains one of these options might look like application.quiet or scripting.allow_multiline_strings, for example.

	application:
		This section contains options that typically control some aspect of the user interface, as well as a few other miscellaneous options that don't really fit anywhere else.

		* as_debug: enables the angelscript debugger (same as -d argument)
		* compilation_message_template = string: allows the user to change the formatting of errors and warnings (see remarks at the bottom of this article)
		* GUI: attempts to print information using message boxes instead of stdout/stderr
		* quiet: no output is printed upon successful compilation (same as -q argument)
		* QUIET: attempts to print as little to stdout and stderr as possible (though this option is still a work in progress)

	build:
		This section contains options that are directly related to the compiling/bundling of an NVGT game into it's final package. It contains everything from options that help NVGT find extra build tools for certain platforms to those that define the name and version of your product.

		* android_home string defaults to %ANDROID_HOME%: path to the root directory of the Android sdk
		* android_install = integer default 1: should Android apks be installed onto connected devices if signing was successful, 0 no, 1 ask, 2 always
		* android_jaava_home string defaults to %JAVA_HOME%: path to the root of a java installation
		* android_manifest string defaults to prepackaged: path to a custom AndroidManifest.xml file that should be packaged with an APK file instead of the builtin template
		* android_path string defaults to %PATH%: where to look for android development tools
		* android_signature_cert = string: path to a .keystore file used to sign an Android apk bundle
		* android_signature_password = string: password used to access the given signing keystore (see remarks at the bottom of this article)
		* linux_bundle = integer default 2: 0 no bundle, 1 folder, 2 .zip, 3 both folder and .zip
		* mac_bundle = integer default 2: 0 no bundle, 1 .app, 2 .dmg/.zip, 3 both .app and .dmg/.zip
		* no_success_message: specifically hides the compilation success message if defined
		* output_basename = string default set from input filename: the output file or directory name of the final compiled package without an extension
		* precommand = string: a custom system command that will be executed before the build begins if no platform specific command is set
		* `precommand_<platform>` = string: allows the execution of custom prebuild commands on a per-platform basis
		* `precommand_<platform>_<debug or release>` = string: allows the execution of custom prebuild commands on a per-platform basis  with the condition of only exicuting on debug or release builds
		* postcommand = string: a custom system command that will be executed after the build completes but before the success message
		* `postcommand_<platform>` = string: allows the execution of custom postbuild commands on a per-platform basis
		* `postcommand_<platform>_<debug or release>` = string: allows the execution of custom postbuild commands on a per-platform basis with the condition of only exicuting on debug or release builds
		* product_identifier=string default com.NVGTUser.InputBasenameSlug: the reverse domain bundle identifier for your application (highly recommended to customize for mobile platforms, see compiling for distribution tutorial)
		* product_identifier_domain = string defaults to com.NVGTUser: everything accept the final chunk of a reverse domain identifier (used only if build.product_identifier is default)
		* product_name=string defaults to input file basename: human friendly display name of your application
		* product_version = string default 1.0: human friendly version string to display to users in bundles
		* product_version_code = integer default (UnixEpoch / 60): an increasing 32 bit integer that programatically denotes the version of your application (default is usually ok)
		* product_version_semantic = string default 1.0.0: a numeric version string in the form major.minor.patch used by some platforms
		* shared_library_excludes = string default "plist TrueAudioNext GPUUtilities systemd_notify sqlite git2 curl": Partial names of shared libraries that should not be copied from NVGT's source lib directory into the bundled product.
		* shared_library_recopy: If this is set, any shared libraries will be copied from scratch instead of only if they're newer than already copied.
		* windows_bundle = integer default 2: 0 no bundle, 1 folder, 2 .zip, 3 both folder and .zip
		* windows_console: when compiling for windows, build with the console subsystem instead of GUI

	scripting:
		This section contains options that directly effect the Angelscript engine, almost always by tweaking a property with the SetEngineProperty function that Angelscript provides.

		The result is undefined if any value is provided outside suggested ranges.

		For more information on these properties, the [Angelscript custom options documentation](https://www.angelcode.com/angelscript/sdk/docs/manual/doc_adv_custom_options.html) is a great resource. There are some engine properties shown on that page which NVGT does not support the configuration of, as a result of such properties being used internally by the engine to function properly.

		* allow_multiline_strings: allow string literals to span multiple lines
		* allow_unicode_identifiers: allow variable names and other identifiers to contain unicode characters
		* allow_implicit_handle_types: experimentally treat all class instance declarations as though being declared with a handle (classtype@)
		* alter_syntax_named_args = integer default 2: control the syntax for passing named arguments to functions (0 only colon, 1 warn if using equals, 2 colon and equals)
		* always_impl_default_construct: create default constructors for all script classes even if none are defined for one
		* compiler_warnings = integer default 0: control how Angelscript warnings should be treated same as -w argument (0 discard, 1 print and continue, 2 treat as error)
		* do_not_optimize_bytecode: disable bytecode optimizations (for debugging)
		* disallow_empty_list_elements: disallow empty items in list initializers such as {1,2,,3,4}
		* disallow_global_vars: disable global variable support completely
		* disallow_value_assign_for_ref_type: disable value assignment operators on reference types
		* disable_integer_division: Defer to floatingpoint division internally even when only integer variables are involved
		* expand_default_array_to_template: cause compilation messages which would otherwise contain something like string\[\] to instead contain array\<string\>
		* heredoc_trim_mode = integer default 1: decide when to trim whitespace of heredoc strings (0 never, 1 if multiple lines, 2 always)
		* ignore_duplicate_shared_interface: allow shared interfaced with the same name to be declared multiple times
		* init_call_stack_size =  integer default 10: the size of the call stack in function calls to initially allocate for each script context
		* init_stack_size =  integer default 4096: the initial stack size in bytes for each script context
		* max_nested_calls = integer default 10000: specify the number of nested calls before a stack overflow exception is raised and execution is aborted
		* max_stack_size = integer default 0 (unlimited): the maximum stack size in bytes for each script context
		* max_call_stack_size = integer default 0: similar to max_nested_calls but can possibly include calls to system functions (angelscript docs is unclear)
		* private_prop_as_protected: private properties of a parent class can be accessed from that class's children
		* property_accessor_mode = integer default 3: control the support of virtual property accessors (0 disabled, 1 only for c++ registrations, 2 no property keyword required, 3 property keyword required)
		* require_enum_scope: access to enum values requires prepending the enum name as in enumname::enumval instead of just enumval
		* use_character_literals: cause single quoted one-character string literals to return an integer with that character's codepoint value

	`#pragma` directives:
		In a few cases, it is also possible to configure some aspects of NVGT's behavior directly from within nvgt scripts themselves using the `#pragma` preprocessor directive.

		This directive is used to safely tell the engine about anything that doesn't directly have to do with your script code but also without causing some sort of compilation error due to bad syntax. A pragma directive could do anything, from embedding a file to selecting assets to choosing  to adding include directories and more.

		The syntax for a pragma directive looks like `#pragma name value` or sometimes just `#pragma name` if the option does not require a value. In some cases when a value is to contain a long or complex enough string such as a path, you may need to surround the value in quotes such as `#pragma name "value."`

	Available directives:
		* `#pragma include <directory>`: search for includes in the given directory (directive can be repeated)
		* `#pragma stub <stubname>`: select what stub to compile using (see remarks at the bottom of this article)
		* `#pragma embed <packname>`: embed the given pack into the compiled executable file itself
		* `#pragma asset <pathname>`: copy the given asset/assets into the bundled product as resources
		* `#pragma document <pathname>`: copy the given asset/assets into the bundled product as documents intended for the user to access rather than the programmer
		* `#pragma plugin <plugname>`: load and activate a plugin given it's dll basename
		* `#pragma compiled_basename <basename>`: the output filename of the compiled executable without the extension
		* `#pragma bytecode_compression <level from 0 to 9>`: controls the compression level for bytecode saved in the compiled executable (0 disabled 9 maximum)
		* `#pragma console`: produce the compiled executable on windows using the console subsystem for CLI based programs

	Remarks on complex options:
		This section contains any explanations of topics that were too bulky to fit in the documentation of each specific configuration option or article section.

	application.compilation_message_template:
		This useful option allows you to control the format that errors and warnings are printed in. The default template looks like this, for example:

		`file: %s\r\nline: %u (%u)\r\n%s: %s\r\n`

		Most specifically, the format string is passed to the function Poco::format in the c++ codebase, and that function receives 5 dynamic arguments. string file, uint line, uint column, string message_type, string message_content.

		It is not needed to understand how this c++ string formatting function works to customize the message template however, you can just reorder the arguments and add text to the following example template:

		`%[1]u %[2]u %[0]s; %[3]s: %[4]s`

		This example moves the line number and column to the beginning of the string, before printing the filename, the message type and the content all on a single line. NVGT automatically adds one new line between each engine message printed regardless of the template.

	platform and stub selection:
		One possible area of confusion might be how the platform and stub directives fit together. In short, the stub option lets you choose various features or qualities included in your target executable while platform determines what major platform (mac, windows) you are compiling for.

		Though explaining how NVGT's compilation process works is a bit beyond the scope of this article, the gist is that Angelscript bytecode is attached to an already compiled c++ program which is used to produce the final executable. There are several versions of the c++ program (we call that the stub) available with more that could appear at any time. For example a stub called upx will produce an executable that has already been compressed with the UPX executable packer, while a stub available on windows called nc will insure that the Angelscript compiler is not included in your target application. In the future we hope to include several more stubs such as one focusing much more on performance over file size, one focusing on minimal dependencies to insure no third party code attributions are needed, etc.

		Internally, you can see this at play by looking in the stub directory of NVGT's installation. You can see several files with the format `nvgt_<platform>_<stubname>.bin`, or sometimes just `nvgt_<platform>.bin` which is the default stub for a given platform. Such files are used to create the final game executable produced by NVGT's compiler, and are selected exactly using the configuration pragmas and options described. If platform is set to windows and stub is set to nc, the file nvgt_windows_nc.bin is used to produce the compiled executable of your game. The only exception is if the platform is set to auto (the default), which will cause your executable to be compiled for the host platform you are compiling on.

	bundle naming and versioning:
		NVGT includes several directives particularly in the build configuration section that allow you to control how your product is identified to end users and to the systems the app is running on. While some are purely for display, others are manditory in certain situations especially when you plan to distribute your app on networks like the app store, google play and similar. For more info on these directives, you should read the "configuring bundling facility" section of the compiling for distribution topic.

	Conclusion:
		Hopefully this tutorial has given you a good idea of how you can get NVGT to perform closer to how you would like in various areas from the user interface to the syntax handling. If you have any ideas as to configuration options you'd like to see added, please don't hesitate to reach out on github either with a discussion or pull request!


	Compiling your project for distribution:
		Once your game is ready to be distributed to others, you may want to turn it into a compiled binary executable, or perhaps even a complete bundle. The compilation procedure assembles all the individual nvgt files of your source code into a single file that can run natively on its targeted operating system without having Nvgt available to interpret it, and the bundling facility can then take that binary along with your sounds/assets, documentation, shared libraries/plugins as well as any other material your game requires and compress/bundle all of that into a complete package ready for you to directly share with the world.

	Here are some thoughts about reasons to do this:
		* Doing this can let you make your work more of a closed-source affair which limits the ease at which someone can reverse engineer your work and hack in changes if that is something you don't want to happen, which is probably the case if your game isn't a free one for sure.
		* The end user will just be able to run the executable file natively, no need to have Nvgt installed. Just like a regular program they download anywhere else!
		* Fewer files, especially if you have a lot of scripts included. That all can turn into just one file, except for the libraries, which can clean up the directory considerably.
		* Stability. No need to worry about whether someone's going to be trying to run your code on an older or newer version of Nvgt with script-breaking changes.

	Some thoughts about reasons not to do this:
		* Your project is open for all to hack and slash at, with the code readily available.
		* You want people to be able to review your work for what ever reason. Since the code is right there, they can read it if they know how and determine what the code does if they so choose.
		* You're creating tutorials or test scripts for others to learn by or to provide examples for how to do something or get feedback, such as trying to find a bug or teaching others how to do something in Nvgt. Hey, that's kinda the same thing as item 2!

	What is an executable file?:
		This will be sometimes called a binary. It's a file that is a complete program directly compatible with the operating system for which it is intended. It can be launched directly, and is usually bundled along side sounds, shared libraries and other data required to run your game.

	What is a bundle?:
		A bundle, unlike a plain executable binary, is instead a full game package which contains everything your game needs to run including any sounds, shared libraries, and the aforementioned executable file. It might be a .zip that the user extracts, a .dmg on MacOS, a .apk file for android and similar for other supported platforms.

	What about all the different operating systems?:
		Since Nvgt is intended to be able to run on Windows, Linux, Mac OS and mobile platforms like Android and iOS, there will be different executables for each targeted operating system. On windows, the resulting binary will have a .exe extension like other programs you may have seen. On other platforms though, there is no standard file extension for executable files and on Android, you never see the executable directly but instead only the final .APK file as there is no reasonable way to run NVGT applications on mobile without fully bundling them first.

		e.g. if you disable the bundling facility and then compile my_game.nvgt on Windows you'll get a my_game.exe but if you compile on linux, you'll just get a file called my_game. Keep this in mind when trying to compile on a non-windows platform. If you have a file called my_game.nvgt and next to it is a directory called my_game, the compilation process will fail because my_game already exists and is a directory!

		When compiling on a non-windows platform, the executable permission bitt on the resulting file is automatically set for you.

	How to compile:
		* on Windows, navigate to the main script file of your code (the one with the void main () function in it). For example it might be called my_game.nvgt
			* Right click the .nvgt file (you can use the context key or shift+f10) and expand the "Compile Script" sub menu. Each platform you'll see there has 2 options:
				* Choose "platform (debug)" if you will want extra debugging information to be included in the program. This can be handy if you're working on it still and want to gain all information about any crashes that may happen that you can get.
				* Choose "platform (release)" if you don't want the extra debug information. You probably should always use release for anything you don't want others to be able to hack and slash at, since the debug information could make it easier on them.
			* If you've disabled the bundling facility, ensure that the necessary library files are located in the same directory your new exe is, or within a lib folder there.
			* If you would rather, you can also run the nvgtw.exe program itself and use the buttons there to compile for any given platform.
		* On Linux, cd to the path containing a .nvgt script.
			* Now run /path/to/nvgt -c scriptname.nvgt where scriptname.nvgt should of course be replaced with the name of your script file. You can select a platform with the -p argument E. -pwindows.
		* On Mac OS, it's best to just click on the NVGT application directly.
			* After launching NVGT you can select compile a script in release/debug mode, choose a platform, and browse to your script file.
			* If, however, you wish to build on MacOS using the command line, cd to the directory containing a .nvgt script and run the command /applications/nvgt.app/Contents/MacOS/nvgt -c scriptname.nvgt, or alternatively open -a nvgt --args -c \`pwd\`/scriptname.nvgt

		You will receive a popup or STDOut print letting you know how long the compilation took. You should find a new file in the same folder you just compiled that has the same name it does but with a different extension e.g. when compiling my_game.nvgt you'll have my_game.exe. Distribute this along with the necessary libraries in what ever form is required. You can set it all up with an installer, such as Inno Setup, or if your game will be portable you can just zip it up, etc.

	Configuring the bundling facility:
		NVGT tries to provide a one-click bundling facility, that is, clicking the compile option on an NVGT script will directly create a package that you can distribute to your end users with little hastle. You don't need to go searching for libraries, you don't need too learn how app bundles on MacOS or Android APK files work, you can just build your game into a distribution ready package.

		While this is a very powerful system that does work out of the box, it likely needs input from you if it is to produce the most optimal results. The facility needs to know where your assets are, it is best if it knows what version your game is / a good display name string for the app (used on various platforms), and various other options allow for anything from custom app manifest files to custom system commands that you can cause to be executed before or after a successful compile.

		Other than including assets like sounds, most inputs to the bundling facility are provided via configuration options. Usually a file called .nvgtrc containing configuration directives is created along side a project, though it can also be called gamename.properties, gamename.json or gamename.ini where gamename is the name of your main nvgt script without the extension. You can learn more about the configuration subsystem specifically in the toolkit configuration tutorial, which includes a list of every available bundling option.

		All of the bundling related options are in the build section of the configuration. So if you want to disable the bundling facility entirely on windows for your project, for example, you might add the line build.windows_bundle = 0 to your .nvgtrc file which would cause a standalone executable to be created when the game is compiled rather than a full bundle. If the bundling facility is not for you at all, you can disable it globally by creating a file alongside the nvgt.exe application itself called config.properties with the same key, more details and alternate configuration filenames are also described in the configuration tutorial.

		Many of the directives here allow you to control how your product is identified to end users and to the systems the app is running on. While some are purely for display, others are manditory in certain situations especially when you plan to distribute your app on networks like the app store, google play and similar.

		If you plan to release your game on MacOS, iOS, or Android, it is highly recommended that you create what's known as a reverse domain identifier for your app by setting the build.product_identifier property. It's called the reverse domain identifier because the format is like a website in reverse, such as com.samtupy.nvgt. In mild cases it might be used to group similar apps together E. the operating system or distribution service can implicitly tell that apps with an identifier starting with com.samtupy all belong to the same company, but in extreme cases this identifier might be used by the system as the main way your app is identified, such as on Android for example where the filesystem paths to the apps data folders actually include this identifier. The identifier can contain periods, uppercase / lowercase letters, and numbers, so long as a number is not the first character following a period. There should be at least 2 period delimited segments in this identifier (prefferably at minimum 3), and it might be a good idea to limit the first segment to popular top level domains like com, org, net etc. While in reality a couple of other characters like dashes and underscores might work in identifiers, they may not be as platform agnostic as the rules listed above. It is not important that the reverse website/app identifier you come up with here actually exists on the internet. If you do not provide this value, the default is com.NVGTUser.scriptname where scriptname is the name of the NVGT file passed to the compiler without the extension. It is possible, however, to only customize the com.NVGTUser part if you want NVGT to use the script filename to derive the rest of the identifier for you. So for example if you know that all games you'll be developing on your computer will be from the company com.epicdevelopers, you can set the key build.product_identifier_domain = com.epicdevelopers in nvgt's global config files. Now when you compile mygame.nvgt, the full reverse domain identifier for that app will be set to com.epicdevelopers.mygame.

		Unfortunately, different operating systems and services use different versioning schemes. Windows and MacOS really want version numbers in major.minor.patch format, in fact the windows version resource actually contains 4 binary words in it's structure to store each period separated version component. While the parsed format isn't actually embedded into a binary structure on MacOS, it's still manditory as soon as you want to distribute your game in the app store, because that service reads such a property from the bundle to identify the version of the app to track updates. As such, the build.product_version_semantic property must be set for MacOS app store distributions, while on windows it is an optional display feature that will cause version information to show for your compiled executable. On the other hand, Android and the google play store use an integer version code to track when a given version of an APK is newer than any installed version. This simple requirement makes it very easy to manage this property automatically on android, it is done by dividing the unix timestamp in seconds by 60, you can also set the build.product_version_code property if you wish to control this manually. Finally, the build.product_version configuration option allows you to set, where possible, the version string that is actually displayed to end users.

		Another configuration option worth mentioning is the build.shared_library_excludes property. This defaults to the string "plist TrueAudioNext GPUUtilities systemd_notify sqlite git2 curl", meaning that by default the shared libraries for plugins are not bundled. As such, you may need to replace this property with your own list of excludes if you use any of NVGT's plugins. We intend to make this step unnecessary, instead determining the list of plugin libraries based on the plugin pragmas contained in the script. So for example if you wanted to include the curl plugin at the moment, you might set the property build.shared_library_excludes = "plist TrueAudioNext GPUUtilities systemd_notify sqlite git2" in your project's configuration file.

		To include assets like sounds and documents into your bundle, the asset and document pragmas are used. A pragma is a type of code statement, similar to `#include,` that allows you to feed NVGT various extra information about your game. For example if one wants to bundle the sounds.dat file with their product, they can add the line of code`#pragma asset sounds.dat` to the top level of one of any .nvgt files that make up the project, or `#pragma document changelog.txt` to include a text file intended for the user rather than the game to access.

		The asset and document pragmas are very similar, on  most platforms indeed they do the same thing. The distinction is that in some cases it can be useful for NVGT to know whether an included asset is intended for access by the game or by the user. The changelog and readme documents for your game should be included as documents rather than assets, for example, because they are included outside the .app folder within a .dmg bundle on MacOS as opposed to assets which are placed in the MacOS/Content/Resources directory, which may be inconvenient for the user to access. Similar distinction is applied on other platforms when required/possible.

		Sometimes you may have a file that you wish to include as an asset, but as a different name in the bundle than that of the file stored on disk. To do this, you can express the asset pragma like `#pragma asset sounds.dat;audio.dat` if you want sounds.dat to actually be called audio.dat in the bundle. The asset and document pragmas will copy/include directories recursively. For example if you have a folder called sounds, the code `#pragma asset sounds` will successfully include the entire sounds folder in your bundle.

	Libraries needed for distribution:
		The following information can be helpful if you have disabled NVGT's bundling facility and you wish to package your compiled games yourself.

		There are a few libraries (dependencies) that Nvgt code relies on. When running from source, Nvgt is just interpreting your .nvgt scripts and using the libraries found in its own installation folder. Generally, if all is normal that will be:
		* On Windows, c:\Nvgt
		* On Linux, where you extracted NVGT's tarball to.
		* On Mac OS, `/Applications/nvgt.app/Contents`

		Within that folder you should have a subfolder that is called "lib" (or "Frameworks" on macOS). Lib contains the redistributable dependencies Nvgt uses for certain functionality. When you run from source, the Nvgt interpreter uses this very folder. However, when you compile, you must provide the files in the same directory as the compiled program so that it can find them. You may copy the entire lib folder to the location of the compiled game, or you can optionally just copy the files directly there if you don't want a lib folder for some reason. In other words, you can have a subdirectory within the program's directory called lib with the files inside, or you can also just place the dlls right in the program's directory where the binary is. On windows, you can omit some of the dll files not needed by your code, and we plan to add more and more support for this on other platforms as time goes on. A list is provided below of the basic purpose and importance of each library so you can hopefully know what to omit if you don't want all of them. These files must be included where ever the program ends up, so installers need to copy those over as well, and they should be included in any zip files for portable programs, etc. Again remember that NVGT usually handles this for you and this step is only required if you wish to create your own bundles by hand.

		At time of writing, the files are:
		* bass.dll is required for the sound object and the tts_voice object, though will not be later once we switch from bass to the miniaudio library. The moment one of these objects exists in your script, your program will crash if this dll is not present.
		* bassmix.dll is just as important as bass.dll and allows things like nvgt's mixer class to exist, where we can combine many bass channels into one. It is always required when bass.dll is used
		* bass_fx.dll is used for reverb, filters, etc and is also required whenever bass.dll is used.
		* nvdaControllerClient64.dll used to speak and Braille through the NVDA screen reader.
		* SAAPI64.dll used to speak and Braille through the System Access screen reader.
		* ZDSRAPI.dll used to speak and Braille through the ZDSR screen reader.
		* git2.dll is the libgit2 library which allows people to programmatically access git repositories (can be good for adding version control to your online game's map world for example). You don't usually need it.
		* git2nvgt.dll is the plugin for nvgt itself that wraps git2. So if your script does not include the line #pragma plugin git2nvgt, then you don't need either of the git2 dlls.
		* nvgt_curl.dll is another plugin that wraps the libcurl library. It used to be the only way to do http requests, but it's now being phased out in favor of more portable options. It's only required if your script includes the line #pragma plugin nvgt_curl
		* nvgt_sqlite.dll is similarly for interfacing with SQLite and is only needed if #pragma plugin nvgt_sqlite is defined.
		* GPUUtilities.dll is used for the GPU acceleration of ray tracing used for geometric reverb in steam audio. It should always be completely optional.
		* phonon.dll is steam audio, that's the HRTF and geometric reverb and cool things like that. You should only need it if you create a sound_environment class or set sound_global_hrtf=true in your script, but at time of writing, sound output using normal sound objects wasn't working as expected without this.
		* TrueAudioNext.dll is more optional GPU acceleration for steam audio.
		* systemd_notify.dll is a plugin wrapping the sd_notify linux function which can be useful if you are writing a systemd service. You probably do not need this otherwise.

		These are organized here roughly in order of most critical to least, but it all depends generally on your project. The first few items are required for the game to even run, but the rest are only necessary if you're using them, and the documentation for using those plugins will ideally make that apparent.

	Detecting library availability:
		Sometimes, you might want to display a friendly error message to your users if a library is unavailable on platforms that allow you to do so rather than the application just crashing. You can do this with a function in NVGT called `bool preglobals()` which executes automatically similar to the main function but before any global variables that might tap into one of these libraries are ever initialized. It can return false to abort the execution of the application. Then, you can use the properties called SOUND_AVAILABLE and SCREEN_READER_AVAILABLE to safely determine if the subsystems could be loaded. The following code snippet would be OK, for example.
		bool preglobals() {
			if (!SCREEN_READER_AVAILABLE) alert("error", "cannot load screen reader components");
			else if (!SOUND_AVAILABLE) alert("error", "cannot load soundsystem");
			else return true; // success
			return false; // one of the libraries failed to load.
		}

	Crediting open source libraries:
		NVGT is an open source project that uses open source dependencies. While aside from bass (which we will replace) we have been very careful to avoid any open source components that forbids or restricts commercial usage in any way, we do not always avoid dependencies that ask for a simple attribution in project documentation such as those under a BSD license. We'd rather focus on creating a good engine which we can fix quickly that can produce some epic games with awesome features, without needing to discard a library that implements some amazing feature just because it's devs simply ask users to do little more than just admit to the fact that they didn't write the library.

		To facilitate the proper attribution of such libraries, a file called 3rd_party_code_attributions.html exists in the lib folder. We highly recommend that you distribute this file with your game in order to comply with all attribution requirements which you can read about in the aforementioned document.

		The idea is that by simply distributing this attributions document within the lib folder of your app, any user who is very interested in this information can easily find and read it, while any normal game player will not stumble upon random code attributions in any kind of intrusive manner. The attributions document is currently well under 100kb, so it should not bloat your distribution.

		The existing file attributes all open source components in NVGT, not just those that require it. If you want to change this, you can regenerate the file by looking in the doc/OSL folder in NVGT's github repository to see how.

		For those who really care about not distributing this file, we hope to provide extra stubs in the future which do not include any components that require a binary attribution. However, there is no denying that this goal is very costly to reach and maintain, while yielding absolutely no true reward for all the time spent sans not needing to distribute an html file with game releases which players and even most developers will not be bothered by anyway. As such, it might be a while before this goal comes to fruition, and we could decide at any point that it is not a viable or worthwhile project to attempt if we find that doing so in any way detriments the development of NVGT, such as by making a feature we want to add very difficult due to lack of available public domain libraries for that feature.


	Concurrency Tutorial:

	Introduction to multithreading and parallelism in NVGT:

		Multithreading, often referred to as concurrency, is the process by which a program can divide tasks into separate, independent flows of execution. These tasks, known as threads, can run concurrently within a single process, allowing for more efficient use of resources and faster execution of complex programs. This approach differs from parallelism, where the tasks are executed simultaneously on multiple processors or computers.

		Concurrency focuses on the structure and design of the program to handle multiple tasks that can overlap in execution. This is particularly useful in scenarios where tasks can be interleaved and do not need to run at exactly the same time, such as managing user interfaces, I/O operations, or handling multiple client requests in a server.

		In contrast, parallelism aims to improve computational speed by executing multiple tasks simultaneously. This can be achieved through multi-core processors, where each core can execute a separate thread, or distributed systems, where tasks are distributed across different machines in a network.

		While NVGT does not support parallelism as one might use it in other programming languages, it does support threads, and this article is all about that. The above distinctions are important, however, because they are quite different, and the way one would go about implementing parallelism is vastly different (and more complicated) than how one would implement concurrency. We denote the distinction explicitly here because often these terms are used interchangeably when they do, in fact, mean very different things, although on a single computer they can appear to be very similar. We will not be discussing parallelism any further in this article, however, as concurrency is the more important aspect here.

	Why is this so important?:

		By far, the easiest way to program is by sticking to synchronous execution, which is what happens by default unless any concurrency is specifically introduced by the programmer. Synchronous execution means that each statement in your code runs to completion before the next is executed. However as your game grows, you may find that it begins to execute slowly, which may temmpt you to quickly implement concurrency into your application.

		This however poses a significant issue because NVGT supports three types of concurrency, and not knowing what method to use or when to use it can often result in slower or more buggy code than if either a different method of concurrency had been used, or if concurrency had not been used at all. People most often jump strait to threads to solve a problem of performance, when, 99 percent of the time, the performance degradation comes from an easily solvable issue in the person's code and not NVGT or a true need for threads or the lowest levels of concurrency. It can be not only bad practice but also detrimental to your program's development to use more advanced concurrency methods when a simple loop would have sufficed, but knowing when and how to spin up a thread when synchronous execution just won't cut it can also save the day in such cases. As such, this article attempts to explain:

		* What exactly concurrency is;
		* How to use concurrency correctly; and
		* The golden rules of concurrency.

		The above explainers are important because concurrency is easy to use improperly. Users of it who don't understand it are bound to make critical mistakes that can cause any number of things, such as:

		* Sequential execution, i.e., absolutely no actual gain and a waste of resources
		* Data races
		* Deadlocks
		* And worse

		They'll also notice their code suffering very strange bugs or exhibiting very odd behaviors, all of which is very, very difficult to debug, let alone track down in the first place, especially on larger projects, and it's best that you be aware of these issues before ever considering threads.

	Three types of concurrency?:

		There are three types of concurrency that NVGT supports: async, coroutines, and threads. We'll discuss all of them in this article.

	Note:

		Any plugins you load may add even more methods of concurrency or even parallelism. Though this article will help you use those extra methods correctly, you should always read the documentation of those plugins to get a firm grasp of how their extra methods work.

	Async:

		Async is the first type of concurrency, and one of the easiest to use. It's also one of the simplest to understand. Although async (may) use a thread (or even multiple) under the hood, it is entirely possible that it may not, and you, as the programmer, needn't care how it works as long as your able to do what needs to be done. In the majority of cases, this is probably the furthest you will ever need concurrency in your game.

		Unlike the other two forms of concurrency, the engine insulates you (mostly) from the problems and mistakes of concurrency that you may make. This is particularly true for functions like `url_get` and `url_post` where the engine can complete the request while you do other things, or in any case that involves writing an asyncronous function that does not share any state with the rest of your program. Take, for example, this code:

		```nvgt
		async<string> result(url_get, "https://nvgt.dev");

		This class is known as a templated class. Though it is beyond the scope of this article, the `string` part is the most important, besides the arguments of the constructor, which specify both the function to be executed and it's arguments. When you create (or instantiate) a class that's templated, NVGT generates the code for that specific class automatically for you, using the types you specify in the angle brackets. This is done on the fly and doesn't harm performance in any manner.

		You may not realize it, but this constructor can take up to 16 parameters. (Woohoo, that's a lot!) When the constructor completes, `result` has some things of interest:

		* The `value` property, which is (in this case) of type `string`, and contains the return result of the function invoked. If you change `string` above to, say, `int`, then this will be of type `int`, not `string`.
		* A `complete` and `failed` property, which tells you if the task has finished executing, or failed in some manner, respectively.
		* An `exception` property which provides you any information that you need if, during the execution of the function, it failed and threw one.
		* A `wait()` method which pauses, or blocks, your code from continuing until the function completes or fails.
		* a `try_wait` function, returning type `bool`, which takes a timeout and will block until either the timeout( also known as the deadline) expires or the task completes.

		As stated previously, this is, for the majority of cases, the only thing you will need to reach for in your toolbox, and it's the highest level of concurrency available. The next two are much lower level, and give you more control, at the cost of raising the steaks by quite a bit. Even then, we strongly recommend reading the sections below about what can go wrong when 2 bits of code running at the same time try accessing the same global variable or bit of memory if you intend to write your own functions that are to be called with this async construct.

	Coroutines:

		A coroutine is a function that suspends itself at certain points. This suspension is called yielding, and using coroutines is known as cooperative multitasking.

		In computer architecture and operating systems, there are generally two types of multitasking approaches. Multitasking, for those who are unsure of the definition (and no, we don't mean multitasking in normal human language), is the concurrent execution or handling of multiple computational processes by a CPU or other processing units within a computer system, such that multiple tasks are performed during overlapping time periods. The key definition is "overlapping time periods." The two types that are generally accepted are known as cooperative multitasking and preemptive multitasking.

		Cooperative multitasking is the first type and works by requiring that a task executes until it reaches a point where it can allow another task to run; this is known as a yield point. When the task reaches this point, it makes a call to a function provided by the operating system, runtime environment, or whatever is running each task, telling it that it is ready to step aside and allow something else to run. In NVGT, this is done through the `yield()` function. Cooperative multitasking relies on each task being well-behaved and voluntarily yielding control, ensuring that all tasks get a chance to execute. This approach is simpler and can be more efficient in environments where tasks can be trusted to yield regularly. However, if a task fails to yield, it can cause the entire system to become unresponsive.

		Preemptive multitasking, on the other hand, is used in all well-known operating systems. Preemptive multitasking works by allocating a task a certain amount of time to run, known as a time slice, and allowing it to run until that time slice ends. When the time slice has elapsed, the operating system forcefully suspends (or "preempts") the execution of the task and replaces it with another task, and the cycle repeats forever. Usually, time slices are very short (perhaps only a few microseconds) and so this gives the illusion that the computer is able to do many things simultaneously. In multicore and multiprocessor systems, this is actually true: the computer can and does do many things all at the same time since the multitasking can occur on all the processors at the same time. However, all that's really happening is that the system is constantly performing this song and dance on all the processors in your computer, all at the same time!

		You may have noticed that we used the word "task" instead of "process" or "thread." This is because, in a computer, there are a few different types of "tasks," and they're all the same to the computer:

		* Process: A fully loaded program that runs. It has code, data, assets, etc.
		* Thread: A task within a process. It shares the code and data of its parent program and is able to access all the state within. More on this later.

		So, how does this have anything to do with coroutines? Well, when you create a coroutine, you are engaging in cooperative multitasking. The coroutine will begin execution as soon as you call `create_coroutine` and will pause the execution of all your other code until you call `yield()`. So, you must remember to yield at some point! The best places to do this are in loops or when your function is about to do something that could take a while, for example, some network-based IO. If you don't, none of your other code can run!

		However, the same applies to the rest of your code: it, too, must yield; otherwise, your coroutine will never be able to proceed!

		Unlike threading, which involves preemptive multitasking, it is (theoretically) impossible for a coroutine that executes in this manner to cause data races or other concurrency problems. This is because only one flow of execution is allowed at any given moment. You can definitely make it appear that multiple things are happening at once if you yield enough, but you'll never be able to duplicate the kind of problems that threading has, so you usually don't need to worry about those. This does not, however, mean that coroutines are the go-to option for all scenarios.

		Coroutines are particularly useful for scenarios where tasks need to perform lengthy operations without blocking the entire program. For instance, they are excellent for handling asynchronous I/O operations, long-running computations that need to provide intermediate results, and any situation where tasks need to cooperate smoothly without complex synchronization mechanisms.

		There are some rules to remember when using coroutines, however:

		* Ensure that your coroutines yield control back to the caller at appropriate points, particularly in loops or before performing lengthy operations.
		* Do not perform blocking operations (like waiting for I/O) within a coroutine without yielding. Instead, yield control and resume the coroutine once the operation completes. The `async<T>` class can help with this: you can store it in a variable your code has access to, and when it completes, you can `yield` to allow one of your coroutines to continue.
		* Always clean up resources before yielding or ensure that resources are properly managed upon resumption. This helps prevent resource leaks and ensures that the coroutine does not hold onto resources unnecessarily.
		* Design your coroutines to work well with others.
		* Thoroughly test your coroutines, particularly their yield and resume behavior. Debugging coroutines can be challenging due to their non-linear execution flow, so use logging and debugging tools to track their state and behavior.

		To create a coroutine, call the `create_coroutine` function:

		```nvgt
		void create_coroutine(coroutine @func, dictionary @args);

		Your coroutine should take the form:

		```nvgt
		void coroutine(dictionary@);

		In both of these, `dictionary@` is a dictionary of arguments that the coroutine needs to execute. When your ready to yield, simply call `yield`:

		```nvgt
		void yield();

		As an example, let's say we wanted to calculate the factors of an integer. Although this is fast for small numbers, it can very quickly become quite computationally expensive when the numbers are particularly large. We might define the coroutine as:

		```nvgt
		void generate_factors(dictionary@ args) {
		    const uint64 number = uint64(args["n"]);
		    uint64[] factors = {1, number}; // Initialize factors with 1 and the number itself.

		    for (uint64 i = 2; i * i <= number; ++i) {
		        if (number % i == 0) {
		            factors.insert_last(i);
		            if (i * i != number) {
		                factors.insert_last(number / i);
		            }
		        }
		        yield(); // Yield control after each iteration
		    }

		    factors.sort_ascending();

		    for (uint64 i = 0; i < factors.length(); ++i) {
		        println("%0".format(factors[i]));
		        yield(); // Yield control after printing each factor
		    }
		}

		Then, before we start our game loop:

		```nvgt
		dictionary args;
		args["n"] = 100000;
		create_coroutine(@generate_factors, @args);

		Now, in our game loop, we need somewhere to yield:

		while (true) {
		    wait(1);
		    yield();
		    ...
		}

	Threads:

	STOP!:

		Before you consider using threads, understand that they are the most complex and error-prone way to handle concurrency. Threads share state with your entire program, meaning they can easily cause data races and deadlocks if not managed correctly. These issues can make your game crash, behave unpredictably, or become very difficult to debug. Unless you have a specific, compelling reason to use threads, stick to `async` (or coroutines if you really need them).

	Note:

		This section is extremely technical in places. This is because multi-threading can be difficult. Even after doing it for a few years you will begin to learn that there are subtle mistakes you can make without realizing it. After we get through the different methods of protecting your code, we'll get into how to use threads and the rules on using them properly. If you want to know how to use any of the synchronization primitives discussed here, the documentation is your friend, as well as the NVGT community.

	Introduction:

		A thread is much different than a coroutine. A thread runs alongside your code, and could even run on other processors in the system. A thread introduces many other problems that coroutines don't: they share state with the rest of your game, meaning they have access to all the global variables the rest of your code does.

		One of the most critical rules of threads is to avoid global or shared state. Global or shared state means any global variables that you have in your code, as well as any state that your thread may access that other threads (also) might access.

		If two or more threads access shared state without any kind of protection, this is known as a data race. A data race occurs when two threads want to perform different operations on a variable, and it just so happens that they do it the same time, or close enough that it doesn't matter. For example, if thread 1 wants to read a sound handle and another thread wants to initialize it, it may just so happen that both operations overlap, causing the handle that thread 1 gets to be in some weird undefined state. Data races can have all kinds of dangerous consequences that could make your program crash, cause unpredictable behavior, and so on. It's even known to cause a write to a variable to mysteriously vanish!

		Data races are also one of the hardest things for a programmer to debug. This is because it can't be predicted when they'll even occur. One run of your code might be fine, but the next run might cause the race, or it might only occur every 200 runs of your code. This is why protection is so important.

	Protecting against data races:

		There are several ways of protecting your code against data races if you do share state:

		* Locks or semaphores
		* Atomic variables
		* Events/condition variables
		* Just don't share any state
		* Message passing (which will be explained here after it is added to NVGT)

	Locks and semaphores:

		A lock is a synchronization primitive used to manage access to a shared resource by multiple threads. When a thread wants to "acquire" the lock, it checks if another thread already holds it. If the lock is already acquired, the requesting thread is blocked until the lock becomes available. When the lock is released, the blocked thread can proceed. This mechanism prevents concurrent access to the resource, ensuring data consistency and integrity.

		Semaphores are another important synchronization mechanism. A semaphore is a signaling mechanism that can be used to control access to a common resource by multiple threads in a concurrent system. Semaphores can be used to solve various synchronization problems, such as controlling access to a finite number of resources or coordinating the order of thread execution. Semaphores maintain a counter representing the number of available resources. Threads can increment the counter to signal the release of a resource or decrement it to wait for a resource. When the counter is zero, any thread attempting to decrement it is blocked until the counter is incremented by another thread.

		There are several types of locks and synchronization mechanisms. One common type is the mutex, or mutual exclusion lock. A mutex ensures that only one thread can access a resource at a time. When a thread acquires a mutex, other threads attempting to acquire the same mutex are blocked until it is released. This is useful for protecting critical sections of code where shared resources are accessed or modified.

		Spinlocks are another type of lock, where the thread repeatedly checks if the lock is available in a loop, or "spins," until it can acquire the lock. Spinlocks are efficient when the wait time is expected to be short, as they avoid the overhead of putting the thread to sleep and waking it up. However, they are not suitable for long wait times, as they consume CPU cycles while spinning.

		Read-write locks, or RW locks, allow multiple threads to read a resource simultaneously but provide exclusive access for writing. This lock has two modes: read mode and write mode. Multiple readers can hold the lock concurrently, but a writer must wait until all readers have released the lock, and vice versa. This is particularly useful in scenarios where read operations are frequent and write operations are infrequent, as it allows for greater concurrency.

		Recursive locks are designed to be acquired multiple times by the same thread without causing a deadlock. They keep track of the number of times they have been acquired by the thread and require the same number of releases to unlock completely. This is useful in scenarios where the same thread might need to re-enter a critical section of code.

		Binary semaphores, also known as mutex semaphores, can have only two states: 0 or 1, representing locked or unlocked. They function similarly to mutexes but do not have ownership, meaning any thread can release them, not just the thread that acquired them. This is useful for simple synchronization tasks where the ownership of the lock is not important.

		Lastly, counting semaphores can have a value greater than one, allowing them to manage a finite number of resources. They maintain a counter representing the number of available resources. Threads increment, or signal, the counter when a resource is released and decrement, or wait, the counter when a resource is acquired. This is useful for limiting access to a pool of resources, such as a fixed number of database connections.

	Wait wait, deadlocks?:

		yeah, did I forget to tell you about those? A deadlock is where two or more threads are unable to proceed with their execution because each is waiting for the other to release a resource they need.

		To understand deadlocks, consider an example where two threads, Thread A and Thread B, need access to two resources, Resource 1 and Resource 2. If Thread A acquires Resource 1 and Thread B acquires Resource 2, both threads might then try to acquire the resource held by the other. Thread A will wait for Resource 2 to be released by Thread B, while Thread B waits for Resource 1 to be released by Thread A. Since neither thread can release the resource it holds until it acquires the resource held by the other, they are both stuck, leading to a deadlock.

		Four necessary conditions for a deadlock to occur are:

		* At least one resource must be held in a non-shareable mode, meaning only one thread can use the resource at any given time.
		* A thread holding at least one resource must be waiting to acquire additional resources currently held by other threads.
		* Resources cannot be forcibly taken from the threads holding them; they must be released voluntarily by the holding thread.
		* There must exist a set of threads such that each thread is waiting for a resource held by the next thread in the set, forming a circular chain.

		Preventing or resolving deadlocks typically involves ensuring that at least one of these conditions cannot hold. Techniques include:

		* Impose a strict order in which resources must be acquired and ensure that all threads adhere to this order.
		* Allow deadlocks to occur, but have mechanisms to detect and recover from them, such as terminating one or more of the threads involved.
		* Use algorithms that dynamically analyze resource allocation requests to ensure that they do not lead to deadlocks, such as the Banker's Algorithm.
		* Implement timeouts for resource requests, forcing a thread to release its held resources if it cannot acquire the required resources within a certain period (this, by far, may be the easiest algorithm to implement).

	Atomic variables:

		An atomic variable is a special type of variable used in concurrent programming to ensure that operations on the variable are completed without interruption. The key characteristic of atomic variables is that any operation performed on them -- such as reading, writing, or modifying -- must be completed entirely before any other operation can start. This means that the state of the variable is always consistent and predictable, even when multiple threads are accessing or modifying it simultaneously.

		To understand this better, consider a scenario where multiple threads are trying to increment a shared counter variable. In a non-atomic scenario, one thread might read the value of the counter, and another thread might read the same value before the first thread has a chance to write the incremented value back, leading to both threads writing back the same value. This causes the counter to be incremented only once instead of twice, leading to incorrect results.

		However, with an atomic variable, the increment operation is indivisible. This means that when one thread starts to increment the counter, no other thread can read or write to the counter until the increment operation is fully completed. The operation is guaranteed to either fully succeed or fail, but it cannot be partially completed. This prevents the problem of race conditions, where the outcome depends on the unpredictable sequence of thread execution.

		Modern processors support atomic operations through special hardware instructions that ensure these operations are performed without interruption. These instructions include atomic read-modify-write operations, such as compare-and-swap (CAS) and fetch-and-add, which are used to implement atomic variables.

		For example, consider the compare-and-swap (CAS) operation. It works by checking if the value of the variable is equal to an expected value. If it is, the value is replaced with a new value. If it isn't, the operation fails. This all happens in one atomic step, ensuring that no other thread can interfere between the check and the update.

	Events/condition variables:

		Events are synchronization primitives used to signal between threads, allowing one thread to notify others about the occurrence of a specific condition. In the context of concurrency, an event can be set or reset, and threads can wait for an event to be set before proceeding. This is particularly useful when you need one or more threads to pause execution until a certain condition is met.

		For instance, imagine a scenario where a worker thread is processing tasks from a queue. The worker thread can wait for an event that gets set when new tasks are added to the queue. Once the event is set, the worker thread wakes up and processes the new tasks. If there are no tasks, the worker thread goes back to waiting. This mechanism ensures efficient use of CPU resources, as the worker thread is not continuously polling the queue but instead waits for a signal.

		Events can be manual-reset or auto-reset. A manual-reset event remains signaled until it is explicitly reset, allowing multiple waiting threads to be released. An auto-reset event automatically resets after releasing a single waiting thread, ensuring that only one thread is woken up per signal.

		Condition variables are another synchronization mechanism used to block a thread until a particular condition is met. They are typically used in conjunction with mutexes. When a thread waits on a condition variable, it releases the associated mutex and enters a waiting state. When the condition is signaled, the mutex is reacquired, and the thread resumes execution.

		The primary use case for condition variables is to manage complex thread interactions that require waiting for certain states or conditions. For example, in a producer-consumer scenario, a consumer thread might wait on a condition variable until there are items available in a buffer. The producer thread, after adding an item to the buffer, signals the condition variable to wake up the consumer thread.

		Here’s a simplified example: a producer and consumer thread are started one after the other. The producer thread (which is what produces data for the consumer thread to act upon) acquires the mutex, adds an item to the buffer, signals the condition variable, and releases the mutex. By contrast, the consumer thread acquires the mutex, waits on the condition variable while the buffer is empty, processes the item from the buffer the condition variable has been signaled, and releases the mutex. The "mutex," in this  case, could (and should most likely be) a recursive mutex.

		Condition variables support two main operations: `wait` and `notify`. The `wait` operation puts the thread into a waiting state and releases the mutex. The `notify_one` operation wakes up one waiting thread, while `notify_all` wakes up all waiting threads. These operations are critical for coordinating complex interactions and ensuring that threads only proceed when the required conditions are met.

		Events are simpler to use when you need to signal one or more threads to proceed. They are particularly effective when dealing with straightforward signaling scenarios, such as waking up worker threads when a new task arrives.

		Condition variables are more suitable for scenarios requiring complex synchronization, especially when multiple conditions and interactions between threads need to be managed. They provide more control over the waiting and signaling process, making them ideal for use cases like producer-consumer problems.

	Not sharing any state:

		If all of the above has left you utterly confused and wondering what you've just read, that's okay; you have lots of other options. The simplest would be to just ignore this section entirely and to just not use threads at all. If you really, really, really do need threads, though, the best method would be to figure out how you don't need to share state. This might be quite difficult in some cases, but the less state you share, the less likely it is you'll need to worry about any of this stuff. If you do need to share state, all of the above is important to understand to do things properly.

	Creating and managing threads:

		To create a thread, call the `thread` constructor and give it a name of your thread. This is usually an ID that you can use to look up the thread later. This will NOT start your thread, though; to do that, call `start` passing in both a function to execute and (optionally) a dictionary handle of arguments that that thread should receive.

		Your thread callback should take as input a `dictionary@` which contains the arguments given in `start()`.

		Once started, a thread immediately begins execution. Your original code that spawned the thread will continue executing after this function returns.

		If you want to wait until the thread completes, call `join()` without any arguments. Alternatively, `join` can take a timeout which, if expires, causes the function to return `false`.

		On the thread object there are a few other things you can do as well, such as changing the threads priority via the `priority` property.

	Warning!:

		Do not change the priority of a thread unless you absolutely have to. Messing around with the thread priority of a thread -- particularly with threads which do a lot of computationally expensive work -- can cause your entire system to hang and have other undesireable side-effects!

	Golden rules of threads:

		When using threads, always remember:

		* Only use threads when you need to. Do not just use them because you feel they would make your code faster. More often than not, this will just cause your code to run slower.
		* When using locks, hold the lock for the shortest time possible. The area in which a lock is acquired is known as a critical section, and you should do only what you neeed to do in these and then immediately release the lock. If you don't do this, you could cause resource leaks, sequential execution, deadlocks and all kinds of other problems.
		* Do not acquire a lock at the start of a function. This may seem tempting, but it is wrong, and will cause your code to be far slower, if not eliminate the benefit of using threads entirely.
		* Avoid recursive mutexes unless you know what you are doing.
		* When acquiring a lock, there are two ways of doing so: `lock` and `try_lock`. If your able to, use `try_lock` which will return false if the lock cannot be acquired. If this does return false, go do something else and try acquiring the lock later.
		* If your code consists of many threads that mostly perform reads of shared state and only a few cases where shared state is modified, use a reader-writer lock and not a mutually exclusive one.
		* Do not mess with thread priorities.
		* The less shared state your threads have to manage, the fewer synchronization issues you'll encounter. Where possible, avoid global variables or shared objects.
		* For simple counters or flags, use atomic operations instead of locks to avoid the overhead and complexity of mutexes.
		* Do not use busy-waiting (spinning) to wait for a lock or condition. This wastes CPU resources and can lead to performance issues. If you need to wait for a lock to be available for acquisition, call `lock` on it and allow the underlying operating system to do the waiting for you.
		* Creating and destroying threads can be expensive. Reuse threads with thread pools or similar techniques to manage resources efficiently.
		* Where possible, use higher-level concurrency constructs like `async<T>`. These abstractions often handle the complex details of synchronization for you.
		* When using condition variables, always protect the condition check with a lock and use a loop to recheck the condition after waking up, as spurious wakeups can occur. That is, it is possible for your thread to be awoken when the condition variable hasn't actually been signaled.
		* While using timeouts can prevent deadlocks, ensure that your program can handle the scenario where a timeout occurs gracefully.
		* Clearly document the synchronization strategy in your code, including which locks protect which data. This helps maintainers understand the concurrency model and avoid introducing bugs.
		* Concurrency bugs can be rare and hard to reproduce. Test your application under load to increase the chances of exposing synchronization issues early in development.

	Conclusion:

		This article provided a deep dive into the various concurrency mechanisms provided by NVGT. Though parts may have been terse, we still hope it helped. If you have any questions, don't hesitate to ask!


	Memory Management Information (incomplete):
		What is a handle? What are references? What is the difference between the stack and the heap and when/why should I care? What is a value typed object? What is a primative datatype, and when should or shouldn't one use the @ character in their code? When should I worry about garbage collection and what even is that? How much is it wise to bother with all this stuff anyway?

		If you've been having any such questions or similar, you've come to the right place. This tutorial will attempt to unravel and demystify all of the jargon behind various memory management techniques and structures as well as advice about when/how to use them. The goal is that by the end of this reading, you will have the knowledge to begin developing confidence particularly in the usage of handles and references in your games, while also learning a lot of interesting stuff about memory management along the way.

	It's automatic, mostly:
		One thing to keep in mind when developing games in NVGT is that most of the memory management is done for you automatically. If you are just getting in to game creation, you don't need to worry about memorising everything you read in this tutorial from the beginning because NVGT will make as many wise memory management decisions for you as it reasonably can, mostly with the use of reference counting based garbage collection (described below). There are certainly times when you specifically need to direct the compiler especially if you notice your game running slowly or being more abusive to ram usage than you would like, but truly you can code even a pretty advanced game in NVGT while only knowing a fraction of the knowledge contained in this tutorial.

	Reference counting:
		Since it's a pretty simple concept though admitedly with a potentially intimidating name and since it's the backbone to most of NVGT's memory management, we'll start by describing the concept of a reference counter.

		The problem: If an object instance is created, how do we know when it is safe to destroy it? We don't want to delete the object before the programmer is done using it, after all.

		The solution: An integer variable within the class called the reference counter stores the number of references your script maintains to an object. When a new variable is created in your script that references an object, it's reference counter increases by 1. When that variable goes out of scope or in some other way gets destroyed/reset, the counter decreases by 1. If the reference counter reaches 0, the object gets destroyed.

		As you may have surmised, this does mean that it is possible to create 2 or more variables that actually point to the same object in memory so long as that object uses the reference counting model. For each variable that exists that points to the same object, the reference counter for that object increases. Such variables in nvgt/Angelscript are called handles, and we'll talk about those below in another section of this tutorial.

		All Angelscript classes created by the NVGT programmer automatically use reference counting based memory management. As in most cases, the internal reference counter is hidden from the user.

		Reference counting is pretty great and simple in many cases, but it has it's drawbacks which cause it to not be viable in all situations, particularly with some of the builtin core types NVGT provides. To understand why, lets define a couple of other low level concepts first.

	The stack:
		Memory allocation can be expensive and slow, and all functions usually require at least a few bytes of memory to do their work, such as to store local variables and other bookkeeping information required for the computer to return to the previous function. If a program were to allocate memory from the system every time a function was called, programs would execute very slowly and RAM modules may meat their end just a bit more quickly. So, how do we execute functions without allocating memory for each call?

		The stack is the name for a chunk of memory allocated on program startup intended specifically to keep track of and manage function calls. Rather than allocating 32 bytes of memory from the system needed by a function every time that function is called, we instead use 32 bytes of the already allocated stack to store data for the function call. The only time the operating system actually needs to perform extra memory allocation is if the stack grows larger than what the system initially allocated to the process, in which case the operating system can perform rare memory allocations to make the stack larger up until the stack reaches the maximum stack size configured by the programmer or allowed by the OS, in which case the program will abort.

		Though various platforms could have slightly different rules to the following, it works something like this. An integer called the stack pointer keeps track of where on the stack data should next be written, a little like the seekable position pointer of a file object. When a function requiring 32 bytes of memory is called, the stack pointer is subtracted by 32 and any byte between the new position of the stack pointer and the old one from before the function call is designated as safe space for the function to store the data it needs to execute. When the function returns, 32 gets added to the stack pointer thus restoring it to it's previous position from before the function call, and any following calls will thus overwrite the stack based memory used by the previous function because instead of reallocating memory over and over from the system per call, we just keep reusing the same already allocated memory over again, which is absolutely spades more efficient. 32 is of course an arbitrary number, all of this magical stack management including determining how much space a function will need on the stack is all done internally for you by the compiler and you rarely to never need to think about these internals, not unless you need to adjust your stack size for some reason or in this case, are trying to learn when reference counted objects and thus handles in Angelscript can and cannot be used.

	The heap:
		Did you notice one of the major drawbacks to the stack mentioned above? As soon as a function that uses the stack to store a variable returns, that memory is, for all intense and purposes, erased / inaccessible. It may be overwritten at any time. So even though the stack is great for avoiding many small memory allocations when performing calculations and calling functions, it can't actually store any long-term data. As you may see, it's only half the puzzle. So how do we solve this problem and establish long-term data storage?

		Enter the heap. It's a memory area like the stack, but with vastly different characteristics. The heap is where most generic in-memory application data is stored. This is because with the heap, the programmer can ask the system to allocate any amount of memory at any time, and then to free/deallocate it at any later time. While it provides a vastly higher amount of flexibility as opposed to the stack (the data you stored on the heap won't disappear as soon as the function that stored it stops executing), it also means that now the programmer is responsible for managing how much memory they allocate and how well they use it. Unless you are very careful, misuse of the heap can make your program run slower than you'd like, or even introduce memory leaks because unless explicitly told, the system will not free any memory allocated on the heap until the entire program exits.

		Remember though that memory allocation, especially in a way where the intricate low level internals of it are hidden from the programmer, is slow. If it wasn't, we might not need structures like the stack to perform fast calculations. Internally it's a complicated, clever and extensive system that manages the heap. A programmer can ask the system to allocate 30mb of memory on the heap and appear to get a nice, contiguous 30mb buffer that they can do what they like with. But internally, the system has potentially done a lot of magic to allocate that 30mb of RAM. Maybe it checked recently freed memory to see if an equal or larger sized block had been recently discarded and could be reused, it might have to locate enough free memory pages to construct a buffer of the given size, on some operating systems compression might even become involved as portions of the 30mb of data remain unaccessed for increasingly longer periods of time. This author certainly does not claim to be an expert on the subject. It's enough to know that because memory allocation onto the heap is actually quite complicated internally, overuse or misuse of it can be slow.

	Combining the stack and the heap:
		So on one hand, we can get really fast access to some memory on the stack, but it's limited and temporary as the memory a function uses is rendered unsafe the moment that the function returns. On the other hand we can get easy access to any amount of memory within reason at any time, but it can be significantly slower and involves mechanisms like reference counting or other techniques to make sure that there are neither memory leaks nor objects getting freed before they should be. There is no perfect solution, thus the existance of both.

		We can use a somewhat flawed analogy to describe these systems. When thinking of the stack, you could imagine a container of disks or books or other flat items, with each item stacked neatly on top of the other. You can either place a new item on the top, or take one off, both of which are very quick operations. That's all you can do though, the container is too tight for you to magically grab the third item down in the stack without upending the entire container, or program as it were. Where as the heap, you could imagine either as a messy room for simplicity or as the most organized library you have ever seen for technicality. While you can both add and retrieve any item at any time, you first need to either sort through the mess on the floor or follow the organization system to the correct shelf to find it, and you never know when less frequently accessed parts of the library might be packed away into boxes to save space.

		So in the end, the stack is where you should perform many small calculations, it's where things like vectors and small strings should be placed, primative function arguments can be passed via the stack after a certain point as well. The heap on the other hand is where either any large hunks of data should be stored, and/or any data where you might need more control over the memory management than what the stack allows for. Both the stack and the heap are very important, but for different reasons.

	Value typed objects:
		So now that we've taken a short jaunt through one of computing's proverbial low level sewer systems that few people want to think about in depth, lets get some badly needed fresh air and return to a much more comfortable higher level armed with some of that acquired knowledge. We'll make the transition by defining one more term.

		Angelscript (the scripting engine NVGT uses) allows the definition/registration of 2 main types of objects internally, or in other words, allows builtin objects to be registered with 1 of 2 memory models. Those are reference typed objects, and value typed objects. A reference typed object is described in various sections regarding reference counting, so for now we'll focus on value typed objects.

		Value typed objects are any type of object which is usually created with memory directly on the stack within a function call, or else directly inlined in memory as part of a class. They are always automatically destroyed as soon as the memory containing them is erased, such as when a function is returned or when a class instance containing them is destroyed. They cannot be created by an nvgt script, and can instead only be registered as builtin objects in NVGT's source code itself. Generally we try to avoid registering value typed objects due to the fact that you cannot make a handle to one of them, but as they are created on the stack rather than the heap, it can be more efficient to register certain object types, such as string, vector, random number generators, and a few other objects as value types as opposed to reference types, which usually require a memory allocation every time one is created. Generally it's best for an object to be a value type if it is likely to exist for a very short period of time. The biggest takeaway here is that you cannot create handles to certain builtin classes, and the reason is because they are value types that will get destroyed as soon as the memory containing them is freed. You can create handles to reference counted objects containing value typed properties, however.

	handles:
		A handle, in short, is a variable that points to any reference counted object instance that has been allocated on the heap. A handle itself is not really an object per say (though it may exist as such on some internal level), it is only a variable that points to one. You can have as many handles to the same object as you want, and every time a handle is created, the reference counter for that object increases. When a handle is destroyed, the reference counter for the object it points to decreases by 1. If the object's reference counter reaches 0, the object is destroyed. Handles themselves are registered as value types, meaning they exist directly on the stack or inlined within the memory of a class. So that means when a class instance containing a handle as a property gets destroyed, that handle contained therein automatically gets destroyed at the same time, causing the reference counter of any object it pointed to to be consequently decremented. Similarly any handles created as variables in a function are destroyed when that function returns, as the handles themselves exist on the stack even though the object the handle points to does not.

		A handle is declared using the @ character following a class type. For example if I want to create a handle to a sound object, I can create a variable like this.
		sound@ snd = null;
		And can then do @snd = sound(); to assign the handle to point to a new sound object at any time or @snd = existing_sound; to similarly cause the handle to point to an existing sound object.

		Hold up though, lets talk about this = null part, because that is important. Remember, a handle is not an object, it is only a pointer to an object. This means that the pointer can actually point at nothing, or null, when it is waiting to be assigned to. It can also be set to null at any time, which will decrement the reference counter of the object it used to point to while keeping the handle around, now not pointing to anything, waiting for reassignment. Why is this important? Well, what happens if we try executing the following code while the snd handle is set to null.
		snd.load("test.ogg");
		We get an unhandled null pointer access exception! The engine does not know what to do in this case because you have instructed it to call the load method on a theoretical sound object, but without giving the engine an actual sound object to call the method on! For this reason, it is always important to insure that an object handle you are about to access is not set to null before you do so. This is particularly important when functions return handles to objects, as the function could decide to return null.
		if (@snd == null) alert("oops", "no sound object associated with this handle");
		else snd.load("test.ogg");
		So what's with the @ character again when checking whether the handle is null? To answer that question, consider this class.
		class example {
			int number;
			bool opEquals(const example@ other) {
				return @other != null and this.number == other.number;
			}
			void opAssign(const example@ other) {
				if (@other != null) this.number = other.number;
			}
		}
		In this case, we have overloaded the == operator for the class, meaning we can execute `if (instance1 == instance2)`. We've also overloaded the assignment operator so that we can do `instance1 = instance2;` to assign the second object to the first.

		The existance of assignment and comparison operators on a class means that when working with handles, we now must distinguish between when we are trying to call the comparison or assignment operator on the class a handle is pointed to, or whether we wish to directly compare or assign one handle to another. Thus, the statement @instance1 = instance2; causes the instance1 handle to point to the instance2 object where as instance1 = instance2; causes instance2 to assign/overwrite instance1 in memory. Similarly if we wish to determine whether a handle is null, we typically must check whether @instance == null, because if we execute instance == null; instead, we're checking whether the object that the instance handle is pointing to is comperable to the value of null by calling it's equality operator, meaning that if the handle is null (not pointing at an object), you'll get the same null pointer exception if you omit the @ character when checking if a handle is null. Angelscript has provided a keyword to get around this ambiguity (at least in the case of comparison) that helps you not need to remember where and where not to put @ characters. You can execute if (instance is null) as an alternative to if (@instance == null), or even if (instance1 is instance2), to compare two handles (meaning checking whether both handles point to  the same object) while insuring that you are never accidentally comparing with the object pointed to by the handle instead.


	Subscripting tutorial:
		Subscripting is the concept of loading external Angelscript code into your game and executing it. This has many applications, including allowing you to make editional levels for your game that aren't directly included, and allowing your game admins to modify how certain items work. This tutorial contains a small example of the basic features of subscripting. We'll go over the ways to load code, find functions, and share methods between the nvgt scripts and your subscripts. For brevity, if I say nvgt code I am talking about the actual .nvgt scripts that are loaded by nvgt.exe and/or compiled. When I say the subscript, I'm talking about the code that the nvgt code is loading (E.G. a shared script in STW).

	Angelscript's Execution Flow:
		This section will talk just a bit about Angelscript's execution flow as it will help make these concepts a bit clearer. You won't specifically need to remember all of this, but it should give you a bit more of a framework for understanding what you do need to know.

		The lowest level of executing an Angelscript is the engine, you don't need to worry about this (I create it in C++). NVGT only uses one Angelscript engine instance for all scripts, be that nvgt code or subscript code. The engine is where we register all of the c++ functions that nvgt can use, for example `key_pressed` or the string class. The only important note is that these functions are registered with different access masks (allowing you to control what functions subscript code have access to), we'll talk about that later.

		The next step of code execution in Angelscript is to tell the engine to start what's known as a module for us. A module is basically a container for code. You feed the module code in as many sections (or files) as you want, then tell the module to build the code, making the module contain compiled bytecode now ready for execution. NVGT itself only uses one module called "nvgt_game", meaning that the "nvgt_game" module contains all the code contained in any .nvgt files that are included. Once the module of code has been compiled, the module is now aware of what functions, classes etc that the code contains. This means we can search for a function by name in a module, then call that function. When subscripting is involved, the nvgt code actually creates a second module at the request of the programmer (you'll see this below). Thus, the subscripting code runs in a different Angelscript module than the nvgt code does, and the most important point of this entire line is that the code in one angelscript module cannot immedietly access code in another module. If you start a new module for subscripting, the code in that module will not automatically have access to the variables or functions in the nvgt_game module, not unless you exclusively share each one.

	Sharing Code:
		There are 2 ways to share functions from the nvgt code to the subscript's code. Both have advantages and disadvantages, and unfortunately, neither are perfect. I'll be the first to admit that sharing functions with subscripts is actually the most annoying part of subscripting by far, though it is the type of thing where you set it up once and it's behind you sans any new functions you wish to share. The following are the considerations:

	Shared code:
		Angelscript does have this concept called shared code. If a module declares something as shared, other modules can easily access it using the "external" keyword in Angelscript. Lets create a small shared class for this example.
		shared class person {
			string name;
			int age;

			person(string name, int age) {
				this.name = name;
				this.age = age;
			}
		}

		Now, so long as the top of our subscripting code contains the line
		`external shared class person;`
		We can successfully create and use instances of the person class in the subscripting code. So this all sounds great, what's the big problem with it? Oh how sad I was when I found out. Basically shared code cannot access non-shared code! Say that the person class shown above had a function in it that accessed a non-shared global variable in the nvgt code called score, the class will now fail to compile because the shared code of the person class cannot access the non-shared variable score. This is a horifying limitation in my opinion that makes the shared code feature much less intuitive to use. A slight saving grace is that at least non-shared code can call shared code E. non-shared code in the nvgt_game module can make instances of the person class just fine, but the person class (being shared code) can't access, for example, the non-shared list of person objects in your game that the person class wants to add instances of itself to. Luckily, there is a much better option in most cases.

	Imported functions:
		Angelscript shared modules provide the concept of importing a function from one module to another. Say you have a non-shared function in your game called `background_update`. If at the top of your subscripting code you include the line
		`import void background_update() from "nvgt_game";`
		and so long as you make one extra function call when building the module with subscripted code in it (`script_module.bind_all_imported_functions()`), the subscripting code can now successfully call the `void background_update()` function even though it is not shared in the nvgt code. The only disadvantage to this system is that it only works for functions! There is no such ability to say, import a class.

		This all means that you will certainly need to combine these 2 methods of sharing code to share all of what you desire. For example you need to use the shared code feature to share a class interface with the subscripting code, but then since shared code can't access non-shared code, you need to use an imported function to actually retrieve an instance of that class (such as the player object) from the nvgt code.

	Full Example:
		Below you'll find a completely categorized example of how subscripting works, incapsilating all the concepts we've discussed thus far.

	Code to be shared with the subscripts:
		// A global array for person objects.
		person@[] people; // Shared code doesn't work with global variables, so you can't share this.

		shared class person {
			string name;
			int age;

			person(const string& in name, int age) {
				this.name = name;
				this.age = age;
			}
		}

		// Say we want the subscripting code to be able to create a new person.
		person@ new_person(const string& in name, int age) {
			person p(name, age);
			people.insert_last(p);
			return p;
		}

		// Or in some cases, this is the closest you'll get to sharing a global variable, so long as it supports handles.
		person@[]@ get_people() property { return @people; }

	Imports:
		Now lets create a section of code that imports the functions. This way, the user who writes subscripting code doesn't have to do it.

		string imports = """import person@ new_person(const string& in, int) from "nvgt_game";
		import person@[]@ get_people() property from "nvgt_game";
		external shared class person;
		""";

	Subscript code:
		string code = """void test() {
			person@ p = new_person("Sam", 21);
			new_person("reborn", -1);
			alert("test", people[0].name);
		}

		// This function needs arguments.
		int64 add(int n1, int n2) {
			return n1 + n2; // normal int can be passed as argument but not yet for return values, consider it a beta bug.
		}

		// This function will be used to demonstrate how to catch script exceptions.
		void throw_exception() {
			throw ("oh no!");
		}
		""";

	Calling the subscript:
	include "nvgt_subsystems.nvgt" // To limit the subscript's access to certain nvgt core functions.:

		void main() {
			// Create a new module.
			script_module@ mod = script_get_module("example", 1); // The second argument can be 0 (only if exists), 1 (create if not exists), and 2 (always create).
			// Add the code sections, usually it's a section per file though they can come from anywhere.
			mod.add_section("imports", imports);
			mod.add_section("code", code);
			// Remember when I talked about an access mask earlier that allows you to limit what core engine functions can be called by the subscript? Now is the time to set that. If the function permissions aren't set at build time, there will be compilation errors and/or wider access than you intended. An access mask is just an integer that is used to store single bit flags as to whether the subscript should have access to a given set of functions. You can see the full list in nvgt_subsystems.nvgt. You can simply binary OR the ones you want to grant access to in this next call, all others will be disabled.
			mod.set_access_mask(NVGT_SUBSYSTEM_SCRIPTING_SANDBOX | NVGT_SUBSYSTEM_UI);
			// Now we need to build the module. We should collect any errors here which are returned in a string array. You should display them if the build function returns something less than 0.
			string[] err;
			if (mod.build(err) < 0) {
				alert("error", join(err, "\r\n\r\n"));
				exit();
			}
			// Next, if any functions are being shared via the imported functions method, we need to bind their addresses from this nvgt_game module to our example module. Don't worry it's just one line + error checking, but what is happening behind the scenes in this next call is that we are looping through all functions that have been imported, and we're searching for them by declaration in the nvgt_game module. When we find them, we tell the example module the function's address.
			if (mod.bind_all_imported_functions() < 0) {
				alert("error", "failed to bind any imported functions");
				exit();
			}
			// Cool, we're ready to go! Everything above this point you can consider an initialization step, code up to this point executes rarely, usually you'll want to store a handle to the script_module object somewhere and other parts of your code will repeatedly perform the steps below as calls are needed. Some wrapper functions that make calling common function types even easier are supplied later in this tutorial. We're about to start calling functions now. We can either retrieve a function by declaration, by index or by name.
			script_function@ func = mod.get_function_by_decl("void test()"); // This is useful because we can limit the returned function by return type and argument types, meaning that if 2 functions with the same name but different arguments exist, you will know you get the correct one.
			if (@func == null) {
				alert("error", "can't find function");
				exit();
			}
			// This looks a bit bulky, but it won't in actual usage when you don't need to show UI upon this error. Lets call the function!
			func({}); // Usually a dictionary of arguments is passed, this function takes none. Soon I think we'll be able to use funcdefs to call functions from shared code completely naturally, but we're not there yet and so now we use a dictionary, properly demonstrated in the next call.
			// Now lets demonstrate how to pass function arguments and how to fetch the return value. We'll skip the error check here we know the add function exists.
			@func = mod.get_function_by_name("add"); // Notice the lack of signature verification here.
			dictionary@ r = func.call({{1, people[0].age}, {2, people[1].age}}); // Notice how we can access those person objects created from the subscript.
			// The return value will be stored in a key called 0, and the other values will maintain the indexes that you passed to them encase the function has modified a value using an &out reference.
			int64 result = 1;
			r.get(0, result);
			alert("add test", result);
			// Usually it's good to check for errors when calling functions, unfortunately. In time this may be compressed so that a default error handler may exist or something of the sort, for now, this is possible.
			err.resize(0);
			@func = mod.get_function_by_name("throw_exception");
			func({}, err);
			if (err.size() > 0) alert("test", join(err, "\r\n"));
		}

	Useful wrapper functions:
		Finally, I'll leave you with some useful functions that show how it's very easy to wrap the calling of common functions. As I said above later I plan to make it possible to call functions with funcdefs if I am able, but we're not yet there. For now, you can use a version of these. They will not run out of the box as they are copied from stw, but they should be all the example needed.

		dictionary shared_function_cache; // It's useful to cache function lookups in a dictionary for performance, else Angelscript needs to keep looping through all shared functions to find the matching signature.

		script_function@ find_shared_function(const string& in decl) {
			script_function@ ret = null;
			if (shared_function_cache.get(decl, @ret)) return @ret;
			@ret = script_mod_shared.get_function_by_decl(decl);
			shared_function_cache.set(decl, @ret);
			return @ret;
		}

		dictionary@ call_shared(const string& in decl, dictionary@ args = null, string[]@ errors = null) {
			script_function@ func = find_shared_function(decl);
			return call_shared(func, args, errors);
		}

		dictionary@ call_shared(script_function@ func, dictionary@ args = null, string[]@ errors = null) {
			if (@func != null) {
				bool internal_error_log = false;
				if (@errors == null) {
					internal_error_log=true;
					@errors = string[]();
				}
				@args = func.call(args, @errors);
				if (@errors != null and internal_error_log and errors.length() > 0)
					log_scripting("error", errors[0], errors.length()>1? errors[1] : "", errors.length()>2? errors[2] : "", "");
			}
			return args;
		}

		void call_shared_void(script_function@ func, dictionary@ args = null, string[]@ errors = null) {
			call_shared(func, args, errors);
		}

		bool call_shared_bool(script_function@ func, dictionary@ args = null, string[]@ errors = null, bool default_value = false) {
			@ args = call_shared(func, args, errors);
			if (@args != null) return dgetb(args, 0);
			return default_value;
		}

		double call_shared_number(script_function@ func, dictionary@ args = null, string[]@ errors = null, double default_value = 0.0) {
			@ args = call_shared(func, args, errors);
			if (@args != null) return dgetn(args, 0, default_value);
			return default_value;
		}

		string call_shared_string(script_function@ func, dictionary@ args = null, string[]@ errors = null, string default_value = "") {
			@ args = call_shared(func, args, errors);
			if (@args != null) return dgets(args, 0);
			return default_value;
		}


API References:
	This section contains several complete class and function API references that allow the creation of games and other applications. To see examples of how any of these objects or functions work together with any great detail, you should look in the NVGT User Manual instead of the API references. However if you are trying to find the name of a specific function or browse the complete list of what this engine has to offer, you have come to the right place!


	Builtin Script API:
		Containers:
			In this documentation, we consider a container to be any generic class that has the primary purpose of storing more than one value.


			Classes:
				array:
					This container stores a resizable list of elements that must all be the same type. In this documentation, "T" refers to the dynamic type that a given array was instanciated with.
					1. `T[]();`
					2. `array<T>();`
					3. `array<T>(uint count);`

				Arguments (3):
					* uint count: The initial number of items in the array which will be set to their default value upon array instanciation.

				Remarks:
					Items in an array are accessed using what's known as the indexing operator, that is, `arrayname[index]` where index is an integer specifying what item you wish to access within the array. The biggest thing to keep in mind is that unlike many functions in NVGT which will silently return a negative value or some other error form upon failure, arrays will actually throw exceptions if you try accessing data outside the array's bounds. Unless you handle such an exception with a try/catch block, this results in an unhandled exception dialog appearing where the user can choose to copy the call stack before the program exits. The easiest way to avoid this is by combining your array accesses with healthy usage of the array.length() method to make sure that you don't access out-of-bounds data in the first place.

					Data in arrays is accessed using 0-based indexing. This means that if 5 items are in an array, you access the first item with `array[0]` and the last with `array[4]`. If you are a new programmer this might take you a second to get used to, but within no time your brain will be calculating this difference for you almost automatically as you write your code, it becomes second nature and is how arrays are accessed in probably 90+% of well known programming languages. Just remember to use `array[array.size() -1]` to access the last item in the array, not `array[array.length()]` which would cause an index out of bounds exception.

					There is a possible confusion regarding syntax ambiguity when declaring arrays which should be cleared up. What is the difference between `array<string> items` and `string[] items` and when should either one be used?

					At it's lowest level, an array is a template class, meaning it's a class that can accept a dynamic type `(array<T>)`. This concept also exists with the grid, async, and other classes in NVGT.

					However, AngelScript provides a convenience feature called the default array type. This allows us to choose just one template class out of all the template classes in the entire engine, and to make that 1 class much easier to declare than all the others using the bracket syntax `(T[])`, and this array class happens to be our chozen default array type in NVGT.

					Therefore, in reality `array<T>` and `T[]` do the exact same thing, it's just that the first one is the semantically correct method of declaring a templated class while the latter is a highly useful AngelScript shortcut. So now when you're looking at someone's code and you see a weird instance of `array<T>`, you can rest easy knowing that you didn't misunderstand the code and no you don't need to go learning some other crazy array type, that code author just opted to avoid using the AngelScript default array type shortcut for one reason or another.

					You can also create an array using braces, like this:
					`array<T> a={item1, item2, item3}`


					Methods:
						back:

							Retrieve the last element in the array.

							T& array::back();

						Returns:

							The last value in the array.

						Remarks:

							This method is functionally equivalent to `array[-1]`.

							An exception will be thrown if the array is empty.

						Example:

							```NVGT
							void main() {
								string[] items = {"Blah", "Bleh", "Blur"};
								alert("Last item", items.back());
							}



						empty:

							Determine whether or not the array is empty.

							1. bool array::empty();

							2. bool array::is_empty();

						Returns:

							bool: true if the array is empty, false if not.

						Remarks:

							This method is functionally equivalent to `array.length() == 0`.

						Example:

							```NVGT
							void main() {
								string[] items;
								bool insert = random_bool();
								if (insert)
									items.insert_last(1);
								alert("The array is", items.empty() ? "empty" : "not empty");
							}



						erase:

							Removes an element of an array by a given position.

							1. void array::remove_at(uint index);

							2. void array::erase(uint index);

						Arguments:

							* uint index: the position to be removed.

						Remarks:

							Please note that the position value is zero-based, and failure to define the correct index will cause NVGT to throw an "index out of bounds" exception.

						Example:

							```NVGT
							void main() {
								string[] names = {"HTML", "CSS"};
								names.remove_at(0);
								alert("Languages", join(names, ", "));
							}



						extend:

							Appends the items from another array into this array.

							void array::extend(const array<T>& arr);

						Arguments:

							* array<T>& arr: The array to be inserted.

						Example:

							```NVGT
							void main() {
								string[] names = {"HTML", "CSS"};
								string[] others = {"JavaScript", "PHP", "ASP"};
								names.extend(others);
								alert("Languages", join(names, ", "));
							}



						find:

							Search for an item in an array.

							1. int array::find(const T&in value);

							2. 	int array::find(uint start_at, const T&in value);

						Arguments:

							* uint start_at: An optional index to begin searching from.

							* const T&in value: The value to search for.

						Returns:

							int: The index of the located item, or -1 if not found.

						Remarks:

							Be ware that this method will start getting slower the larger your array is, this is because this find method must search through every item in the array until it finds an item that equals what you are looking for.

							Be sure to never write code like `string value = array[array.find("testing")];` or anything else where you call array.find() directly within the array's indexing operator. This is because array.find() could return -1, meaning that an index out of bounds exception will be thrown if you use the value returned by this function without verifying it first.

						Example:

							```NVGT
							void main() {
								string text = "this is a sentence made up of many words, sort of?";
								string[]@ elements = text.split(" ,?.", false);
								int word = elements.find("sort");
								if (word < 0) alert("Oh no not found", "Someone should probably report this if they ever see it while running an unmodified version of this example...");
								else alert("found", "The word sort is at index " + word);
							}



						find_by_ref:

							Search for an object handle in an array.

							1. int array::find_by_ref(const T&in value);

							2. 	int array::find_by_ref(uint start_at, const T&in value);

						Arguments:

							* uint start_at: An optional index to begin searching from.

							* const T&in value: The value to search for.

						Returns:

							int: The index of the located item, or -1 if not found.

						Remarks:

							Be ware that this method will start getting slower the larger your array is, this is because this find method must search through every item in the array until it finds an item that equals what you are looking for.

							Be sure to never write code like `string value = array[array.find("testing")];` or anything else where you call array.find() directly within the array's indexing operator. This is because array.find() could return -1, meaning that an index out of bounds exception will be thrown if you use the value returned by this function without verifying it first.

						Example:

							```NVGT
							void main() {
								string text = "this is a sentence made up of many words, sort of?";
								string[]@ elements = text.split(" ,?.", false);
								int word = elements.find_by_ref("sort"); // This should fail, as the string used to construct the word "sort" here is different to the actual string object holding the same word in the array.
								if (word < 0) alert("Not found", "We only find references here.");
								else alert("found... How did that happen?", "The word sort is at index " + word);
							}



						front:

							Retrieve the first element in the array.

							T& array::front();

						Returns:

							The first value in the array.

						Remarks:

							This method is functionally equivalent to `array[0]`.

							An exception will be thrown if the array is empty.

						Example:

							```NVGT
							void main() {
								string[] items = {"Blah", "Bleh", "Blur"};
								alert("First item", items.front());
							}



						insert:

							Inserts an element or another array into an array at a given position, moving other items aside to make room if necessary.

							1. void array::insert(uint index, const T&in value);

							2. void array::insert(uint index, const array<T>& arr);

							3. void array::insert_at(uint index, const T&in value);

							4. void array::insert_at(uint index, const array<T>& arr);

						Arguments (1, 3):

							* uint index: the position to insert at.

							* const T&in value: the element to be inserted.

						Arguments (2, 4):

							* uint index: the position to insert at.

							* array<T>& arr: The array to be inserted.

						Remarks:

							This method adds an element to the given position of an array. If you pass index 0, the item will be inserted at the beginning of the aray, and if you pass the array's absolute length, the item will be inserted at the end. Any values less than 0 or greater than the array's absolute length will result in an index out of bounds exception being thrown.

						Example:

							```NVGT
							void main() {
								string[] names = {"HTML", "CSS"};
								string[] others = {"JavaScript", "PHP"};
								names.insert_at(2, others);
								names.insert_at(3, "ASP");
								alert("Languages", join(names, ", "));
							}



						insert_last:

							Appends an element to an array.

							1. void array::insert_last(const T&in element);

							2. void array::push_back(const T&in element);

						Arguments:

							* const T&in element: the element to be inserted.

						Remarks:

							This method will add the specified element to the end of the array.

						Example:

							```NVGT
							void main() {
								string[] names = {"HTML", "CSS"};
								names.insert_last("Java Script");
								alert("Languages", join(names, ", "));
							}



						length:

							Returns the number of items in the array.

							1. uint array::length();

							2. uint array::size();

						Returns:

							uint: the number of items in the array.

						Remarks:

							This method returns the absolute number of elements in the array, that is, without taking 0-based indexing into account. For example an array containing {1, 2, 3} will have a length of 3, but you must access the last element with the index 2 and the first element with index 0, rather than using 3 for the last item and 1 for the first.

						Example:

							```NVGT
							void main() {
								int[] items;
								for (uint i = 0; i < random(1, 10); i++)
									items.insert_last(i);
								alert("The array contains", items.length() + " " + (items.length() == 1 ? "item" : "items"));
							}



						random:

							Returns a random element in the array.

							1. const T& array::random() const;

							2. const T& array::random(random_interface@ rng) const;

						Remarks:

							This method can be called without a parameter, in which case it will use the default random algorithm, with any of the several built-in random number generators, or any class that implements the random_interface.

						Example:

							```NVGT
							void main() {
								string[] items = {"This", "is", "a", "test"};
								alert("Random word", items.random());
							}



						remove_last:
							Removes the last item from the array.

							1. `void array::remove_last();`
							2. `void array::pop_back();`

						Remarks:
							Calling this method on an empty array will throw an index out of bounds exception.


						remove_range:

							Removes a group of items from an array, starting at a given  position and removing the specified number of items after it.

							void array::remove_range(uint start, uint count);

						Arguments:

							* uint start: the index to start removing at.

							* uint count: the number of items to remove.

						Example:

							```NVGT
							void main() {
								string[] items = {"there", "are", "a", "few", "items", "that","will", "disappear"};
								alert("The array currently is", join(items, ", "));
								items.remove_range(1, 3);
								alert("The array is now", join(items, ", "));
							}



						reserve:
							allocates the memory needed to hold the given number of items, but doesn't initialize them.

							`void array::reserve(uint length);`

						Arguments:
							* uint length: the size of the array you want to reserve.

						Remarks:
							This method is provided because in computing, memory allocation is expensive. If you want to add 5000 elements to an array and you call array.insert_last() 5000 times without calling this reserve() method, you will also perform nearly 5000 memory allocations, and each one of those is a lot of work for the OS. Instead, you can call this function once with a value of 5000, which will perform only one expensive memory allocation, and now at least for the first 5000 calls to insert_last(), the array will not need to repetitively allocate tiny chunks of memory over and over again to add your intended elements. The importance of this cannot be stressed enough for large arrays, using the reserve() method particularly for bulk array inserts can literally speed your code up by hundreds of times in the area of array management.


						resize:
							Resizes the array to the specified size, and initializes all resulting new elements to their default values.

							`void array::resize(uint length);`

						Arguments:
							* uint length: how big to resize the array to

						Remarks:
							If the new size is smaller than the existing number of elements, items will be removed from the bottom or the end of the aray until the array has exactly the number of items specified in the argument to this method.


						reverse:

							Reverses the array, so the last item becomes the first and vice versa.

							void array::reverse();

						Remarks:

							This method reverses the array in place rather than creating a new array and returning it.

						Example:

							```NVGT
							void main() {
								string[] items = {"This", "is", "a", "test"};
								alert("The array is currently", join(items, ", "));
								items.reverse();
								alert("The reversed array is", join(items, ", "));
							}



						shuffle:

							Shuffles the contents of the array.

							1. void array::shuffle();

							2. void array::shuffle(random_interface@ rng);

						Remarks:

							This method can be called without a parameter, in which case it will use the default random algorithm, with any of the several built-in random number generators, or any class that implements the random_interface.

						Example:

							```NVGT
							void main() {
								string[] items = {"This", "is", "a", "test"};
								items.shuffle();
								alert("Garble", join(items, " "));
							}



						sort:
							Sorts an array according to a custom algorithm.

							`void array::sort(array::less&in, uint startAt = 0, uint count = uint(-1));`

						Arguments:
							* array::less&in: A callback that will be used for comparing items.
							* uint startAt: The index to start sorting from (default is 0).
							* uint count: The number of elements to sort (default is -1, meaning all elements will be sorted).

						Remarks:
							The callback function to define is implemented as:
							`funcdef bool array<T>::less(const T&in a, const T&in b);`
							where `t` refers to the array's datatype.

							It should return true if A is considered less than B, or false otherwise.

							Please note the signature must match exactly - parameters must be declared as `const &in` otherwise it will result in a compilation error.


						sort_ascending:
							Sorts an array in ascending order.

							`void array::sort_ascending();`
							`void array::sort_ascending(uint startAt, uint count);`

						Arguments (2):
							* uint startAt: The index to start sorting from.
							* uint count: The number of elements to sort.

						Remarks:
							To sort an array of objects, it is necessary for the class to overload the comparison operator by implementing opCmp.


						sort_descending:
							Sorts an array in descending order.

							`void array::sort_descending();`
							`void array::sort_descending(uint startAt, uint count);`

						Arguments (2):
							* uint startAt: The index to start sorting from.
							* uint count: The number of elements to sort.

						Remarks:
							To sort an array of objects, it is necessary for the class to overload the comparison operator by implementing opCmp.


					Operators:
						opAssign:

							Overloads the = operator, allowing you to assign an array to another, already existing one.

							array& opAssign(const array&in);

						Arguments:

							* const array&in: the array to assign.

						Example:

							```NVGT
							void main() {
								string[] a = {"Apple", "Banana", "Cherry"};
								string[] b = a;
								alert("Fruity!", join(b, ", "));
							}



						opEquals:

							Overloads the == operator, checking if two arrays are equal.

							bool array::opEquals(const array &in other);

						Arguments:

							* const array&in other: the array to compare.

						Example:

							```NVGT
							void main() {
								string[] a = {"Apple", "Banana", "Cherry"};
								string[] b = a;
								if (a == b) alert("Equal", "We have two identical arrays. Hurray.");
							}



						opFor:

							Overloads the OpFor* loop iterators, allowing the foreach loop using a key and a value.

						Example:

							```NVGT
							void main() {
								string[] a = {"Apple", "Banana", "Cherry"};
								foreach(string v, int k: a) {
									alert("Array", "Index "+k+"="+v);
								}
							}



						opIndex:

							Overloads the index operator, allowing you to access its elements using square brackets.

						Example:

							```NVGT
							void main() {
								string[] a = {"Apple", "Banana", "Cherry"};
								alert("Array", "Index 1="+a[1]);
							}



				dictionary:
					This container stores multiple pieces of data of almost any type, which are stored and referenced by unique keys which are just arbitrary strings.
					1. `dictionary();`
					2. `dictionary({{"key1", value1}, {"key2", value2}});`

				Arguments (2):
					* {{"key", value}}: default values to set in the dictionary, provided in the format shown.

				Remarks:
					The idea behind a dictionary, or hash map / hash table as they are otherwise called, is that one can store some data referenced by a certain ID or key, before later retrieving that data very quickly given that same key used to store it.

					Though explaining the details and guts of how hash maps work internally is beyond the scope of this documentation, here is an [overly in depth wikipedia article](https://en.wikipedia.org/wiki/Hash_table) that is sure to teach you more than you ever wished to know about this data structure. Honestly unless you wish to write your own dictionary implementation yourself in a low level language and/or are specifically worried about the efficiency of storing vast amounts of similar data which could stress the algorithm, it's enough to know that the dictionary internally turns your string keys into integer hashes, which are a lot faster for the computer to compare than the characters in the key strings themselves. Combine this with clever use of small lists or buckets determined by even more clever use of bitwise operations and other factors, and what you're left with is a structure that can store thousands of keys while still being able to look up a value given a key so quickly it's like magic, at least compared to the least efficient alternative which is looping through your thousands of data points and individually comparing them in order to find just one value you wish to look up.

					NVGT's dictionaries are unordered, meaning that the get_keys() method will likely not return a list of keys in the order that you added them. In cases where this is very important to you, you can create a string array along side your dictionary and manually store key names yourself and then loop through that array instead of the output of get_keys() when you want to enumerate a dictionary in an ordered fassion. Note, of course, that this makes the entire system less efficient as now deleting or updating a key in the dictionary requires you to loop through your string array and find the key that needs deleting or updating, and dictionaries exist exactly to avoid such an expensive loop, though the same efficiency is preserved when simply looking up a key in the dictionary without writing to it, which can be enough in many cases. While in most cases it's best to write code that does not rely on the ordering of items in a dictionary, this at least provides an option to choose between ordering and better efficiency, should you really need such a thing.

					Look at the individual documented methods of this class for examples of it's usage.


					Methods:
						delete:

							This method deletes the given key from the dictionary.

							1. bool dictionary::delete(const string &in key);

							2. bool dictionary::erase(const string &in key);

						Arguments:

							* const string &in key: the key to delete.

						Returns:

							bool: true if the key was successfully deleted, false otherwise.

						Example:

							```NVGT
							void main() {
								dictionary data;
								data.set("nvgt", "An audiogame engine using AngelScript");
								alert("Information", data.delete("nvgt")); //true
								alert("Information", data.delete("nvgt")); //false because the key is already deleted, another word, it does not exist.
							}



						delete_all:
							This method deletes all keys from the dictionary.

							1. `bool dictionary::delete_all();`
							2. `bool dictionary::clear();`


						exists:

							Returns whether a given key exists in the dictionary.

							bool dictionary::exists(const string &in key);

						Arguments:

							* const string &in key: the key to look for.

						Returns:

							bool: true if the key exists, false otherwise.

						Example:

							```NVGT
							void main() {
								dictionary data;
								data.set("nvgt", "An audiogame engine using AngelScript");
								alert("Information", data.exists("nvgt")); //true
								alert("Information", data.exists("gameengine")); //false
							}



						get:

							Gets the data from a given key.

							bool dictionary::get(const string &in key, ?&out value);

						Arguments:

							* const string &in key: the key to look for.

							* ?&out value: the variable for the value to be associated, see remarks.

						Returns:

							bool: true if the dictionary retrieves the value of the key, false otherwise.

						Remarks:

							Please note that the value parameter should be a variable to receive the value. This means that dictionary will write to the variable, not read from.

						Example:

							```NVGT
							void main() {
								dictionary data;
								data.set("nvgt", "An audiogame engine using AngelScript");
								string result;
								if (!data.get("nvgt", result))
									alert("Error", "Failed to retrieve the value of the key");
								else
									alert("Result is", result);
							}



						get_keys:

							Returns a string array containing the key names in the dictionary.

							string[]@ dictionary::get_keys();

						Returns:

							string[]@: the  list of keys in the dictionary on success, 0 otherwise.

						Example:

							```NVGT
							void main() {
								dictionary data;
								data.set("nvgt", "An audiogame engine using AngelScript");
								string[]@ keys = data.get_keys();
								alert("Keys", join(keys, ", "));
							}



						get_size:

							Returns the size of the dictionary.

							1. uint dictionary::get_size();

							2. uint dictionary::size();

						Returns:

							uint: the size of the dictionary, another word, the total keys on success, 0 otherwise.

						Example:

							```NVGT
							void main() {
								dictionary data;
								data.set("nvgt", "An audiogame engine using AngelScript");
								alert("Dictionary size is", data.get_size());
							}



						is_empty:

							Returns whether the dictionary is empty.

							1. bool dictionary::is_empty();

							2. bool dictionary::empty();

						Returns:

							bool: true if the dictionary is empty, false otherwise.

						Example:

							```NVGT
							void main() {
								dictionary data;
								data.set("nvgt", "An audiogame engine using AngelScript");
								alert("Information", data.is_empty()); //false
								data.delete_all();
								alert("Information", data.is_empty()); //true
							}



						serialize:
							Serializes the dictionary into a string.

							`string dictionary::serialize();`

						Arguments:
							None.

						Remarks:
							The format used for serializing is different to that used by BGT.

							Please note the corresponding deserialize method is a standalone global function.


						set:
							Sets the data into a given key.

							`void dictionary::set(const string &in key, const ?&in value);`

						Arguments:
							* const string &in key: the key to use.
							* const ?&in value: the value to set into this key.

						Remarks:
							If the key already exists in the dictionary, its value will be overwritten.


					Operators:
						opIndex:
							Set or retrieve the value of a dictionary by its key.

							`int opIndex(* const string &in key);`

						Arguments:
							* const string &in key: the key to use.

						Remarks:
							If the key doesn't exist in the dictionary, it will be created, similar to using the set method.

							If the key already exists, its value will be overwritten.

							A value can be set as follows:

							my_dictionary["key"]=value;

							A value can be retrieved, but requires explicit casting, as follows:

							string string_value=string(my_dictionary["key_for_string_value"]);
							int int_value=int(my_dictionary["key_for_int_value"]);
							obj@ object_value=cast<obj>(my_dictionary["key_for_object_value"]);


				grid:

					This type is essentially just a more convenient 2d array. One place it's super useful is when representing a game board.

					In this documentation, "T" refers to the dynamic type that a grid was instanciated with.

					1. `grid<T>();`

					2. `grid<T>(uint width, uint height);`

				Arguments (2):

					* uint width: the width of the grid.

					* uint height: the height of the grid.

				Remarks:

					One of the things that makes this class especially convenient is its opIndex overload. It's possible to index into a grid like:

					`my_grid[1, 2];`

					to access the value at (1, 2) on your grid.

					Besides the constructors listed above, you can also make a grid using the brace syntax, similar to arrays, like this:

					`grid<T> g={{item1, item2}, {item3, item4}}`

				Example:

					```NVGT
					void main() {
						grid<bool> game_board(10, 10); // a 10x10 grid of booleans.
						game_board[4, 4] = true; // set the center of the board to true.
						alert("The center of the board is", game_board[4, 4]);
					}



					Methods:
						height:

							Returns the current height of the grid.

							uint grid::height();

						Returns:

							uint: the height  of the grid.

						Example:

							```NVGT
							void main() {
								grid<int> g(random(1, 10), random(1, 10));
								alert("Grid height is", g.height());
							}



						resize:

							Resize a grid to the given width and height.

							void grid::resize(uint width, uint height);

						Arguments:

							* uint width: the width you want to resize the grid to.

							* uint height: the height you want to resize the grid to.

						Example:

							```NVGT
							void main() {
								grid<int> g(5, 5);
								alert("Original width and height", g.width() + ", " + g.height());
								g.resize(random(1, 100), random(1, 100));
								alert("New width and height", g.width() + ", " + g.height());
							}



						width:

							Returns the current width of the grid.

							uint grid::width();

						Returns:

							uint: the width of the grid.

						Example:

							```NVGT
							void main() {
								grid<int> g(random(1, 10), random(1, 10));
								alert("Grid width is", g.width());
							}



					Operators:
						opIndex:

							Get or set the value of a grid cell given an x and a y.

							T& grid::opIndex(uint row, uint column);

						Arguments:

							* uint row: the row (x value) to check.

							* uint column: the column (y value) to check.

						Returns:

							T&: a reference to the value at the given position. It is of whatever type your grid holds.

						Remarks:

							This function can throw index out of bounds errors, exactly like arrays can, so be careful.

						Example:

							```NVGT
							void main() {
								grid<int> the_grid;
								the_grid.resize(5, 5);
								for (uint i = 0; i < 5; i++) {
									for (uint j = 0; j < 5; j++) {
										the_grid[i, j] = random(1, 100);
									}
								}
								alert("Info", "The center is " + the_grid[2, 2]);
							}



			Functions:
				deserialize:
					Deserializes the contents of a string into a dictionary.

					`dictionary@ deserialize(const string&in);`

				Arguments:
					* const string&in: The string to deserialize.

				Remarks:
					The format used for serializing/deserializing is different to that used by BGT.

					The corresponding `serialize` function can be found as a method in the dictionary class.


		Datatypes:
			In this documentation, we consider a datatype to be any class or primitive that typically stores one value. While such a datatype could contain a pointer/handle to a more complex type that may store many values (in which case the handle itself is the single value the datatype contains), the types documented here do not directly contain more than one piece of data be that a string of text, a dynamically typed handle or just a simple number.

			Beware that a few of these datatypes may get quite advanced, and you are not expected to understand what all of them do (particularly the dynamically typed ones) during the course of normal game development. Instead, we recommend learning about basic datatypes here, and then coming back and looking at any that seemed confusing at first glance when you are trying to solve a particular problem usually relating to a variable you have that needs to be able to store more than one kind of value.


			any:

				A class that can hold one object of any type, most similar to a dictionary in terms of usage but with only one value and thus no keys.

				1. any();

				2. any(?&in value);

			Arguments (2):

				* ?&in value: The default value stored in this object.

			Remarks:

				Though NVGT uses Angelscript which is a statically typed scripting language, there are sometimes occasions when one may not know the type of value that could be stored in a variable, or may want to create a function that can accept an argument of any type. While this isn't the only way to do so, the any object provides a safe way to accomplish this task, possibly the safest / least restrictive of all available such options. There should be no type that this object can't store.

				The biggest downside to this class is simply that storing and retrieving values from it is just as bulky as the usual way of doing so from dictionary objects. That is, when retrieving a value one must first create a variable then call the any::retrieve() method passing a reference to that variable.

				something to note is that the constructor of this class is not marked explicit, meaning that if you create a function that takes an any object as it's argument, the user would then be able to directly pass any type to that argument of your function without any extra work.

				Again just like with dictionaries, there is no way to determine the type of a stored value. Instead if one wants to print the value contained within one of these variables, they must carefully attempt the retrieval with different types until it succeeds where the value can then be printed.

			Example:

				```NVGT
				// Lets create a version of the join function that accepts an array of any objects and supports numbers, strings, and vectors.
				string anyjoin(any@[]@ args, const string&in delimiter) {
					if (@args == null) return ""; // Nobody should ever be passing the null keyword to this function, but if they do this line prevents an exception.
					string[] args_as_strings;
					for (uint i = 0; i < args.length() && args[i] != null; i++) {
						// We must attempt retrieving each type we wish to support.
						int64 arg_int;
						double arg_double;
						string arg_string;
						vector arg_vector;
						if (args[i].retrieve(arg_string)) args_as_strings.insert_last(arg_string);
						else if (args[i].retrieve(arg_vector)) args_as_strings.insert_last("(%0, %1, %2)".format(arg_vector.x, arg_vector.y, arg_vector.z));
						else if (args[i].retrieve(arg_double)) args_as_strings.insert_last(arg_double);
						else if (args[i].retrieve(arg_int)) args_as_strings.insert_last(arg_int);
						else args_as_strings.insert_last("<unknown type>");
					}
					return join(args_as_strings, delimiter);
				}
				void main() {
					string result = anyjoin({"the player has", 5, "lives, has", 32.97, "percent health, and is at", vector(10, 15, 0)}, " ");
					alert("example", result);
				}



				Methods:
					retrieve:
						Fetch the value from an any object and store it in a variable.

						`bool retrieve(?&out result);`

					Arguments:
						* ?&out result: A variable that the value should be copied into.

					Returns:
						bool: true if successful, false if the variable supplied is of the wrong type.

					Example:
						See the main any chapter where this function is used several times.


					store:

						Store a new value in an any object, replacing an existing one.

						void store(?&in value);

					Arguments:

						* ?&in value: The value that should be stored (can be any type).

					Example:

						```NVGT
						void main() {
							int number;
							string text;
							any example;
							example.store("hello");
							example.retrieve(text); // Check the return value of the retrieve function for success if you are not certain of what type is stored.
							example.store(42); // The previous text value has now been deleted.
							example.retrieve(number);
							alert(text, number);
						}



			ref:

				A class that can hold an extra reference to a handle of any type.

				1. ref();

				2. ref(const ?&in handle);

			Arguments (2):

				* const ?&in handle: The handle that this ref object should store at construction.

			Remarks:

				In Angelscript, a handle is the simplest method of pointing multiple variables at a single object, or passing an object to a function without copying it in memory. If you know any c++, it is sort of like a c++ shared pointer. You can only create a handle to a complex object, and a few built-in NVGT objects do not support them such as the random number generators because they are registered as value types. Describing handles much further is beyond the scope of this reference chapter, but you can [learn more about handles from the Angelscript documentation](https://www.angelcode.com/angelscript/sdk/docs/manual/doc_script_handle.html) including why a few complex objects in NVGT don't support them.

				Usually handles must be typed to the object they are pointing at, for example I could create a variable called `dictionary@ d;~ to create an empty handle to a dictionary object.

				The type restriction on handles is almost always perfectly ok, however in a few super rare cases it can become bothersome or could make a coding task more tedious. The ref object is a workaround for that. Where ever typed handles can be used to point to an object, a ref object could point to it as well.

				Unlike normal typed handles, one must cast a ref object back to the type it is storing in order to actually call methods or access properties of the stored type.

				The = (assignment) operator is supported which causes the ref object to point to the value supplied, and the == (equality) operator is supported to compare whether a ref object and either another ref object or typed handle are pointing to the same actual object.

			Example:

				```NVGT
				// Lets create a function that can set the volume of either a mixer or a sound object.
				void set_volume(ref@ obj, int volume) {
					mixer@ m = cast<mixer@>(obj);
					if (@m != null) {
						m.set_volume(volume);
						return;
					}
					sound@ s = cast<sound@>(obj);
					if (@s != null) s.set_volume(volume);
				}
				void main() {
					sound my_sound;
					my_sound.load("c:\\windows\\media\\ding.wav");
					mixer mix;
					my_sound.set_mixer(mix);
					set_volume(mix, -5);
					alert("mixer volume", mix.volume);
					set_volume(my_sound, -10);
					alert("sound volume", my_sound.volume);
					my_sound.close();
				}



			string:
				Methods:
					count:

						Count the number of times a substring appears in another string.

						uint64 string::count(const string&in search, uint64 start = 0);

					Arguments:

						* const string&in search: the substring to search for.

						* uint64 start = 0: the zero-based position to start searching at.

					Returns:

						uint64: the number of occurances of the given substring found.

					Example:

						```NVGT
						void main() {
							alert("test", "testtest".count("test"));
						}



					empty:

						Check to see if a string is empty.

						1. bool string::empty();

						2. bool string::is_empty();

					Returns:

						bool: True if the string is empty; false otherwise.

					Example:

						```NVGT
						void main() {
							string test = input_box("String is Empty Tester", "Enter a string.");
							if (test.empty())
								alert("String is Empty Tester", "The string appears to be empty.");
							else
								alert("String is Empty Tester", "The string isn't empty.");
						}



					ends_with:

						Determines if a string ends with another string.

						bool string::ends_with(const string&in str);

					Arguments:

						* const string&in str: the string to look for.

					Returns:

						bool: true if the string ends with the specified search, false if not.

					Example:

						```NVGT
						void main() {
							string suffix = "test";
							string text = input_box("Test", "Enter a string");
							if (text.is_empty()) {
								alert("Info", "Nothing was entered.");
								exit();
							}
							if (text.ends_with(suffix))
								alert("Info", "The entered string ends with " + suffix);
							else
								alert("Info", "The entered string does not end with " + suffix);
						}



					erase:

						Remove a portion of a string, starting at a given index.

						void string::erase(uint pos, int count = -1);

					Arguments:

						* uint pos: the position to start erasing from.

						* int count = -1: the number of characters to erase after the specified index. If -1, erases all characters starting at the given pos until the end of the string.

					Remarks:

						Please note this modifies the calling string. To keep the original you will need to make a copy.

						In the event of boundary errors (start + count >= length), an exception is thrown.

					Example:

						```NVGT
						void main() {
							string text = input_box("String", "enter at least a 3-character string.");
							if (text.length() < 3) {
								alert("Info", "Your string must be at least three characters.");
								exit(1);
							}
							alert("Before removal, the string is", text);
							text.erase(2);
							alert("After removal, the string is", text);
						}



					escape:

						Modify a string with suitable escape sequencing.

						string string::escape(bool strict_JSON = false);

					Arguments:

						* strict_JSON: Use JSON escaping. Defaults to false if not specified.

					Returns:

						string: A new string with the escaped data.

					Remarks:

						The strict_JSON parameter is only necessary under special circumstances, namely the escape behaviour for characters \x7 and \xb. If you know you won't be touching these characters, you can merely call escape without a parameter and it will still escape safely.

					Example:

						```NVGT
						void main() {
							string test="test\xbtest";
							alert("Escaped", test.escape(false));
							alert("Escaped", test.escape(true));
						}



					find:
						Find the position of a substring from within a string.

						`int string::find(const string &in search, uint start = 0);`

					Arguments:
						* search: The string to search for.
						* start: The index to start searching from. Default is 0 (beginning).

					Returns:
						The position where the substring starts if found, or -1 otherwise.

					Remarks:
						If the start is less than 0, -1 is returned.


					find_first:
						Find the position of a substring from within a string.

						`int string::find_first(const string &in search, uint start = 0);`

					Arguments:
						* search: The string to search for.
						* start: The index to start searching from. Default is 0 (beginning).

					Returns:
						The position where the substring starts if found, or -1 otherwise.

					Remarks:
						If the start is less than 0, -1 is returned.

						This appears to be an alias for string::find.


					find_first_not_of:
						Find the first occurrence of any character that is not present in the search string.

						`int string::find_first_not_of(const string &in search, uint start = 0);`

					Arguments:
						* search: The characters to exclude in a match.
						* start: The index to start searching from. Default is 0 (beginning).

					Returns:
						The position where a match was found, or -1 otherwise.

					Remarks:
						This can be useful in several scenarios, such as splitting strings or quickly finding out if your string has invalid characters. A scenario where this could be useful is if you are finding parts of a URL or path, you might want to `find_first_not_of(":\\/.")`.


					find_first_of:
						Find the first occurrence of any character in the search string.

						`int string::find_first_of(const string &in search, uint start = 0);`

					Arguments:
						* search: A string of characters to search for.
						* start: The index to start searching from. Default is 0 (beginning).

					Returns:
						The position where a match was found, or -1 otherwise.

					Remarks:
						Where find and find_first will find an occurrence of an entire substring, this will match any character. This is useful, for instance, if you need to search for delimiters before splitting.


					find_last:
						Find the position of a substring from within a string, searching right to left.

						`int string::find_last(const string &in search, uint start = -1);`

					Arguments:
						* search: The string to search for.
						* start: The index to start searching from. Default is -1 (end).

					Returns:
						The position where the substring starts if found, or -1 otherwise.

					Remarks:
						This is useful if what you need to find is closer to the end of a string than the start.

						This appears to be an alias for string::rfind.


					find_last_not_of:
						Find the first occurrence of any character that is not present in the search string, searching from right to left.

						`int string::find_last_not_of(const string &in search, uint start = 0);`

					Arguments:
						* search: The characters to exclude in a match.
						* start: The index to start searching from. Default is -1 (end).

					Returns:
						The position where a match was found, or -1 otherwise.

					Remarks:
						This can be useful in several scenarios, such as splitting strings or quickly finding out if your string has invalid characters. A scenario where this could be useful is if you are finding parts of a URL or path, you might want to `find_last_not_of(":\\/.")`.


					find_last_of:
						Find the first occurrence of any character in the search string, searching right to left.

						`int string::find_last_of(const string &in search, uint start = 0);`

					Arguments:
						* search: a string of characters to search for.
						* start: The index to start searching from. Default is -1 (end).

					Returns:
						The position where a match was found, or -1 otherwise.

					Remarks:
						Where find and find_last will find an occurrence of an entire substring, this will match any character. This is useful, for instance, if you need to search for delimiters before splitting.


					format:

						Formats a string with placeholders.

						string string::format(const ?&in = null...);

					Arguments:

						* const ?&in = null: Any stringable value. You can pass up to 16 parameters.

					Returns:

						Returns the formatted string.

					Remarks:

						This method is useful in situations when:

						1. You want to have multiple formats for messages, such as short and long messages.

						2. You don't want to use concatenation.

						3. You want to use a placeholder more than once in a string.

					Example:

						```NVGT
						void main() {
							string[] greetings = {"Hello %0! How are you?", "Hola %0! ¿Cómo estás?"};
							string[] messages = {"You have %0 health", "%0 health"};
							alert("Greeting:", greetings[random(0, 1)].format("Literary"));
							alert("Message:", messages[random(0, 1)].format(1000));
						}



					insert:
						Insert a string into another string at a given position.

						`void string::insert(uint pos, const string&in other);`

					Arguments:
						* uint pos: the index to insert the other string at.
						* const string&in other: the string to insert.

					Remarks:
						Please note this modifies the calling string. To keep the original you will need to make a copy.

						If pos is out of bounds, an exception is thrown.


					is_alphabetic:

						Checks whether a string contains only alphabetical characters and nothing else.

						bool string::is_alphabetic(string encoding = "UTF8");

					Arguments:

						* string encoding = "UTF8": The encoding to check against, with the UTF8 encoding being cached for speed.

					Returns:

						bool: true if the input string contains only alphabetic characters, false otherwise.

					Remarks:

						This function only returns true if the string contains only alphabetical characters in the given encoding. If a single other non-alphabetical character is encountered, this function will return false.

						If this function is called on a string that contains data that is not in the specified encoding, the results are undefined.

					Example:

						```NVGT
						void main() {
							alert("example", "aabdslf".is_alphabetic()); // Should show true.
							string input = input_box("example", "enter a string");
							if(input.empty()) {
								alert("Info", "You must type a string.");
								exit(1);
							}
							if(input.is_alphabetic())
								alert("example", "you typed a string that contains only alphabetical characters");
							else
								alert("example", "this string contains more than just alphabetical characters");
						}



					is_alphanumeric:

						Checks whether a string contains only alphabetical or numeric characters and nothing else.

						bool string::is_alphanumeric(string encoding = "UTF8");

					Arguments:

						* string encoding = "UTF8": The encoding to check against, with the UTF8 encoding being cached for speed.

					Returns:

						bool: true if the input string contains only alphabetic or numeric characters, false otherwise.

					Remarks:

						This function only returns true if the string contains only alphabetical or numeric characters in the given encoding. If a single other non-alphanumeric character is encountered, this function will return false.

						If this function is called on a string that contains data that is not in the specified encoding, the results are undefined.

					Example:

						```NVGT
						void main() {
							alert("example", "aabdslf".is_alphanumeric()); // Should show true.
							string input = input_box("example", "enter a string");
							if(input.empty()) {
								alert("Info", "You must type a string.");
								exit(1);
							}
							if(input.is_alphanumeric())
								alert("example", "you typed a string that contains only alphanumeric characters");
							else
								alert("example", "this string contains more than just alphanumeric characters");
						}



					is_digits:

						Checks if a string is only digits (numbers).

					Arguments:

						* string encoding = "UTF8": The encoding to check against, with the UTF8 encoding being cached for speed.

					Returns:

						bool: true if the input string is only digits; false otherwise.

					Remarks:

						This function only returns true if the string contains digits in the given encoding. If a single non-numaric character is encountered, this function will return false.

						If this function is called on a string that contains data that is not in the specified encoding, the results are undefined.

					Example:

						```NVGT
						void main() {
							alert("Example", "123".is_digits()); // Should return true.
							string input = input_box("Example", "Enter a string");
							if (input.is_digits()) alert("Example", "This string is only digits.");
							else alert("Example", "This input contains non-numaric characters.");
						}



					is_lower:

						Checks whether a string is completely lowercase.

						bool string::is_lower(string encoding = "UTF8");

					Arguments:

						* string encoding = "UTF8": The encoding to check against, with the UTF8 encoding being cached for speed.

					Returns:

						bool: true if the input string is lowercase.

					Remarks:

						This function only returns true if the string contains only lowercase characters in the given encoding. If a single uppercase or punctuation character is encountered, this function will return false.

						If this function is called on a string that contains data that is not in the specified encoding, the results are undefined.

					Example:

						```NVGT
						void main() {
							alert("example", "HELLO".is_lower()); // Should show false.
							string input = input_box("example", "enter a string");
							if(input.is_empty())
								exit();
							if(input.is_lower())
								alert("example", "you typed a lowercase string");
							else
								alert("example", "you did not type a lowercase string");
						}



					is_punctuation:

						Checks whether a string contains only punctuation characters and nothing else.

						bool string::is_punctuation(string encoding = "UTF8");

					Arguments:

						* string encoding = "UTF8": The encoding to check against, with the UTF8 encoding being cached for speed.

					Returns:

						bool: true if the input string contains only punctuation characters.

					Remarks:

						This function only returns true if the string contains only punctuation characters in the given encoding. If a single alphanumeric or other non-punctuation character is encountered, this function will return false.

						If this function is called on a string that contains data that is not in the specified encoding, the results are undefined.

					Example:

						```NVGT
						void main() {
							alert("example", ".?!".is_punctuation()); // Should show true.
							string input = input_box("example", "enter a string");
							if(input.is_empty())
								exit();
							if(input.is_punctuation())
								alert("example", "you typed a string that contains only punctuation characters");
							else
								alert("example", "this string contains more than just punctuation");
						}



					is_upper:

						Checks whether a string is completely uppercase.

						bool string::is_upper(string encoding = "UTF8");

					Arguments:

						* string encoding = "UTF8": The encoding to check against, with the UTF8 encoding being cached for speed.

					Returns:

						bool: true if the input string is uppercase.

					Remarks:

						This function only returns true if the string contains only uppercase characters in the given encoding. If a single lowercase or punctuation character is encountered, this function will return false.

						If this function is called on a string that contains data that is not in the specified encoding, the results are undefined.

					Example:

						```NVGT
						void main() {
							alert("example", "HELLO".is_upper()); // Should show true.
							string input = input_box("example", "enter a string");
							if(input.is_empty())
								exit();
							if(input.is_upper())
								alert("example", "you typed an uppercase string");
							else
								alert("example", "you did not type an uppercase string");
						}



					length:

						Returns the length (how large a string is).

						1. uint string::length();

						2. uint string::size();

					returns:

						uint: The length of a string.

					Remarks:

						Note: This returns the length in bytes, rather than characters of any given text encoding.

					Example:

						```NVGT
						void main() {
							string test = input_box("String Length Tester", "Enter a string to see its length.");
							if (test.length() <= 0)
								alert("String Length Tester", "You appear to have provided an empty string!");
							else if (test.length() == 1)
								alert("String Length Tester", "You appear to have only provided a single character!");
							else
								alert("String Length Tester", "The provided string is " + test.length()+" characters!");
						}



					lower:

						Converts a string to lowercase, returning a new string.

						string string::lower();

					Returns:

						string: the specified string in lowercase.

					Example:

						```NVGT
						void main() {
							string text = input_box("Text", "Enter the text to convert to lowercase.");
							if (text.empty()) {
								alert("Info", "You typed a blank string.");
								exit(1);
							}
							alert("Your string lowercased is", text.lower());
						}



					lower_this:

						Converts a string to lowercase, modifying the calling string object.

						string& string::lower();

					Returns:

						string&: a reference to the specified string in lowercase.

					Example:

						```NVGT
						void main() {
							string text = input_box("Text", "Enter the text to convert to lowercase.");
							if (text.empty()) {
								alert("Info", "You typed a blank string.");
								exit(1);
							}
							text = text.lower_this();
							alert("Your string lowercased is", text);
						}



					remove_UTF8_BOM:

						Remove the UTF-8 BOM from a string.

						string string::remove_UTF8_BOM();

					Arguments:

						None.

					Returns:

						None.

					Remarks:

						A BOM (Byte Order Mark) is a special marker used at the beginning of a text file to indicate the byte order of multi-byte character encodings, such as UTF-16 or UTF-32. Byte order refers to how multi-byte values are stored in memory, for example, little-endian (least significant byte first) or big-endian (most significant byte first).

						While UTF-8 does not require a BOM due to its self-synchronizing, byte-based encoding, some older applications rely on it to detect that a file is encoded in UTF-8.

						In UTF-8, the BOM is represented as the byte sequence \xEF\xBB\xBF. If present, it appears at the very beginning of the string or file.

						Note: This modifies the calling string. If you need to preserve the BOM for later, you will need to make a copy of the string before processing or else reconstruct it manually.

					Example:

						```NVGT
						void main() {
							string test="\xEF\xBB\xBFtest";
							test.remove_UTF8_BOM();
							alert("No BOM", test);
						}



					replace:
						Try to find any occurrences of a particular string, and replace them with a substitution in a given string object, returning a new string.

						`string string::replace(const string&in search, const string&in replacement, bool replace_all = true);`

					Arguments:
						* const string&in search: the string to search for.
						* const string&in replacement: the string to replace the search text with (if found).
						* bool replace_all = true: whether or not all occurrences should be replaced, or only the first one.

					Returns:
						string: the specified string with the replacement applied.


					replace_characters:
						Replace a group of characters with another group of characters, returning a new string.

						`string string::replace_characters(const string&in character_list, const string&in replace_with);`

					Arguments:
						* character_list: The list of characters to replace.
						* replace_with: The characters to replace them with.

					Returns:
						string: A new string with the replacements applied.

					Remarks:
						Characters are replaced in sequence: The first character in the list is replaced with the first character in the replace string, the second with the second and so on. Thus, `"ladies".replace_characters("ld", "bb")` will return "babies".

						The comparison is case-sensitive. `"ladies".replace_characters("LD", "bb")` will still return "ladies".

						If the replacement string is shorter than the character list, the remaining characters in the list will be erased.

						If the replacement string is longer than the character list, excess characters are ignored.


					replace_characters_this:
						Replace a group of characters with another group of characters, modifying the calling string.

						`string& string::replace_characters(const string&in character_list, const string&in replace_with);`

					Arguments:
						* character_list: The list of characters to replace.
						* replace_with: The characters to replace them with.

					Returns:
						string: A reference to the calling string with the replacements applied.

					Remarks:
						Characters are replaced in sequence: The first character in the list is replaced with the first character in the replace string, the second with the second and so on. Thus, `"ladies".replace_characters("ld", "bb")` will return "babies".

						The comparison is case-sensitive. `"ladies".replace_characters("LD", "bb")` will still return "ladies".

						If the replacement string is shorter than the character list, the remaining characters in the list will be erased.

						If the replacement string is longer than the character list, excess characters are ignored.


					replace_range:
						Replace a specified number of characters at the provided index with a new string.

						`string string::replace_range(uint start, int count, const string&in replacement);`

					Arguments:
						* uint start: The position to insert the new string.
						* int count: The number of characters to replace.
						* const string&in replacement: the string to insert.

					Returns:
						string: the specified string with the replacement applied.

					Remarks:
						Note the word "replace" here implies the deletion of an old string and adding a new string in its place. A better, though still slightly ambiguous word might be "overwrite". It is similar, though by no means identical, to the following:
						string.erase(start, count);
						string.insert(start, replacement);

						In the event of boundary errors (start + count >= length), an exception is thrown.

						If count <= 0, no processing takes place.


					replace_this:
						Try to find any occurrences of a particular string, and replace them with a substitution in a given string object, modifying the calling string.

						`string& string::replace_this(const string&in search, const string&in replacement, bool replace_all = true);`

					Arguments:
						* const string&in search: the string to search for.
						* const string&in replacement: the string to replace the search text with (if found).
						* bool replace_all = true: whether or not all occurrences should be replaced, or only the first one.

					Returns:
						string&: a two-way reference to the specified string with the replacement applied.


					resize:

						Resize a string to a particular size (in bytes).

						void string::resize(uint new_size);

					Arguments:

						* uint new_size: the new size of the string (in bytes).

					Example:

						```NVGT
						void main() {
							string test = "abcdef";
							alert("The string is", test.length() + " characters");
							test.resize(3);
							alert("The string is", test.length() + " characters");
						}



					reverse:

						Reverse a string.

						string string::reverse(string encoding = "UTF8");

					Arguments:

						* string encoding = "UTF8": The string's encoding, with UTF8 cached for speed.

					Returns:

						string: The specified string in reverse.

					Remarks:

						This function reverses the characters of a string based on a given encoding. To reverse the raw bytes of a string, for example if you are operating on binary data, see the reverse_bytes function instead.

					Example:

						```NVGT
						void main() {
							string text = input_box("Text", "Enter some text to reverse.");
							if (text.is_empty()) {
								alert("Info", "nothing was entered.");
								exit();
							}
							string result = text.reverse();
							alert("Reversed string", result);
						}



					reverse_bytes:

						Reverses the raw bytes contained within a string.

						string string::reverse_bytes();

					Returns:

						string: A copy of the string with it's bytes reversed.

					Remarks:

						If you want to reverse the characters within a string given an encoding such as UTF8, use the reverse method instead.

					Example:

						```NVGT
						void main() {
							string raw = hex_to_string("1a2b3c4d");
							raw = raw.reverse_bytes();
							alert("reverse_bytes", string_to_hex(raw)); // Should show 4D3C2B1A.
							// Lets make it clear how this function differs from string::reverse. We'll make a string of emojis and show the results of both methods.
							string emojis = "🦟🦗🐜🐝🐞🦂🕷";
							alert("emojis reversed properly", emojis.reverse()); // string::reverse takes the UTF8 encoding into account.
							alert("broken emojis", emojis.reverse_bytes()); // This string no longer contains valid character sequences, and so the emoticons will most certainly not show correctly. Aww you can see if you run this example that it seems even string::reverse_bytes() can't quite get rid of mosquitos though... 😀
						}



					rfind:
						Find the position of a substring from within a string, searching right to left.

						`int string::rfind(const string &in search, uint start = -1);`

					Arguments:
						* search: The string to search for.
						* start: The index to start searching from. Default is -1 (end).

					Returns:
						The position where the substring starts if found, or -1 otherwise.

					Remarks:
						This is useful if what you need to find is closer to the end of a string than the start.


					slice:

						Extracts a substring starting at the given index and ending just before the index end (exclusive).

						string string::slice(int start = 0, int end = 0);

					Arguments:

						* int start = 0: The index where the substring begins (inclusive).

						* int end = 0: The index where the substring ends (exclusive).

					Returns:

						string: A new string with the characters modified.

					Remarks:

						This can be a tripping point due to two factors: Strings are zero-based indexing, and slicing is end-exclusive. Therefore, to retrieve characters 4 to 6, your parameters would be 3, 6.

						Negative indexes count backwards from the end of a string. Thus, "hello".slice(-4, 4) is the same as "hello".slice(1, 4): Both would return "ell".

						If start is greater than the string's length, or negative beyond the start of the string, the result is an empty string.

						If end is 0 or otherwise beyond the string's bounds, it is treated as string.length().

						Any situation where start >= end after resolving indexes returns an empty string. An example of this would be "hello".slice(-3, 1), which would be equivalent to "hello".slice(2, 1).

						This method is useful for extracting a string between two points (for instance when parsing HTML or XML, as in slice(tag_start, tag_end)). For length-based extraction, prefer substr.

					Example:

						```NVGT
						void main() {
							string text = "Hello, world!";
							string substring = text.slice(0, 5);
							alert("Info", substring);
							text = "Welcome to NVGT";
							substring = text.slice(11, 16);
							alert("Info", substring);
						}



					split:

						Splits a string into an array at a given delimiter.

						string[]@ string::split(const string&in delimiter, bool full = true, bool allow_blanks = false);

					Arguments:

						* const string&in delimiter: the delimiter to split at.

						* bool full: Use all or any of the delimiter characters.

						* bool allow_blanks: Specifies whether to allow empty fields.

					Returns:

						string[]@: a handle to an array containing each part of the split string.

					Remarks:

						* The full parameter decides how to treat delimiter strings with multiple characters (such as "\r\n"). If enabled, the entire string is treated as a single delimiter ("\r\n"). If disabled, any character in the string counts as a delimiter ("\r" or "\n"). If the delimiter string is a single character this parameter is effectively meaningless.

						* The allow_blanks parameter decides how to treat consecutive delimiters. See the example for how this works in practice.

						* If the input string is empty, the output depends on the allow_blanks parameter. If true, an array with the blank string as the only element is returned. If false, an empty array is returned.

					Example:

						```NVGT
						void main() {
							// basic split/join example
							string test = "welcome to the example";
							string[]@ parts = test.split(" ");
							alert("Parts", join(parts, ", "));
							// Blank fields test
							string test2="Humphrey,Eccleston,,50";
							alert("New string", test2);
							parts=test2.split(",", true, false);
							alert("Fields with data", parts.length()); // Should output 3.
							parts=test2.split(",", true, true);
							alert("Total fields", parts.length()); // Should output 4.
						}



					starts_with:

						Determines if a string starts with another string.

						bool string::starts_with(const string&in str);

					Arguments:

						* const string&in str: the string to look for.

					Returns:

						bool: true if the string starts with the specified search, false if not.

					Example:

						```NVGT
						void main() {
							string prefix = "abc";
							string text = input_box("Test", "Enter a string");
							if (text.is_empty()) {
								alert("Info", "Nothing was entered.");
								exit();
							}
							if (text.starts_with(prefix))
								alert("Info", "The entered string starts with " + prefix);
							else
								alert("Info", "The entered string does not start with " + prefix);
						}



					substr:

						Extracts a substring starting at the given index and containing a specific number of characters.

						string string::substr(uint start = 0, int count = -1);

					Arguments:

						* uint start = 0: The index where the substring begins (inclusive).

						* int count = -1: The number of characters to include in the substring (see remarks).

					Returns:

						string: A new string with the characters modified.

					Remarks:

						If the count is out of bounds, the string is returned from the start index to the end.

						If the start index is out of bounds, an empty string is returned.

						This is useful for working with fixed length data (for instance, give me 4 characters at position 4). If you want to split a string between start and end points, prefer slice.

					Example:

						```NVGT
						void main() {
							string text = "Hello, world!";
							string substring = text.substr(7, 5); // Extracts "world"
							alert("Info", substring);
						}



					trim_whitespace:

						Trims any and all whitespace from the beginning and end of a string, returning a new string.

						string string::trim_whitespace();

					Returns:

						string: the specified string, with any trailing/leading whitespace removed.

					Remarks:

						Any and all whitespace, including the tab character and new lines, will be removed.

					Example:

						```NVGT
						void main() {
							string text = input_box("String", "Enter a string");
							if (text.is_empty()) {
								alert("Info", "You didn't enter a string.");
								exit();
							}
							string result = text.trim_whitespace();
							alert("Trimmed string", result);
						}



					trim_whitespace_left:

						Trims any and all whitespace from the left side of a string, returning a new string.

						string string::trim_whitespace_left();

					Returns:

						string: the specified string, with the whitespace removed from the left side.

					Remarks:

						Any and all whitespace, including the tab character and new lines, will be removed.

					Example:

						```NVGT
						void main() {
							string text = input_box("String", "Enter a string");
							if (text.is_empty()) {
								alert("Info", "You didn't enter a string.");
								exit();
							}
							string result = text.trim_whitespace_left();
							alert("Trimmed string", result);
						}



					trim_whitespace_left_this:

						Trims any and all whitespace from the left side of a string, modifying the calling string.

						string& string::trim_whitespace_left();

					Returns:

						string&: a two-way reference to the specified string, with the whitespace removed from the left side.

					Remarks:

						Any and all whitespace, including the tab character and new lines, will be removed.

					Example:

						```NVGT
						void main() {
							string text = input_box("String", "Enter a string");
							if (text.empty()) {
								alert("Info", "You didn't enter a string.");
								exit();
							}
							text = text.trim_whitespace_left();
							alert("Trimmed string", text);
						}



					trim_whitespace_right:

						Trims any and all whitespace from the right side of a string, returning a new string.

						string string::trim_whitespace_right();

					Returns:

						string: the specified string, with the whitespace removed from the right side.

					Remarks:

						Any and all whitespace, including the tab character and new lines, will be removed.

					Example:

						```NVGT
						void main() {
							string text = input_box("String", "Enter a string");
							if (text.is_empty()) {
								alert("Info", "You didn't enter a string.");
								exit();
							}
							string result = text.trim_whitespace_right();
							alert("Trimmed string", result);
						}



					trim_whitespace_right_this:

						Trims any and all whitespace from the right side of a string, modifying the calling string.

						string& string::trim_whitespace_right();

					Returns:

						string&: a two-way reference to the specified string, with the whitespace removed from the right side.

					Remarks:

						Any and all whitespace, including the tab character and new lines, will be removed.

					Example:

						```NVGT
						void main() {
							string text = input_box("String", "Enter a string");
							if (text.empty()) {
								alert("Info", "You didn't enter a string.");
								exit();
							}
							text = text.trim_whitespace_right();
							alert("Trimmed string", text);
						}



					trim_whitespace_this:

						Trims any and all whitespace from the beginning and end of a string, modifying the calling string.

						string& string::trim_whitespace_this();

					Returns:

						string&: a reference to the specified string, with any trailing/leading whitespace removed.

					Remarks:

						Any and all whitespace, including the tab character and new lines, will be removed.

					Example:

						```NVGT
						void main() {
							string text = input_box("String", "Enter a string");
							if (text.empty()) {
								alert("Info", "You didn't enter a string.");
								exit();
							}
							text = text.trim_whitespace_this();
							alert("Trimmed string", text);
						}



					unpacket:
						Copies individual raw bytes from the string (starting at the given offset) into the provided output variables.

						`int string::unpacket(int offset, ?&out var1, ...);`

					Arguments:
						* offset: Position in the string to start extracting from.
						* var1: The first variable slot to fill.

					Returns:
						A new offset in the string where you should continue the extraction process.

					Remarks:
						The "..." above denotes that this function can take a variable number of arguments. In fact, it can take up to 16 output variables.

						The following snippet will save the string "hello" into five variables named a, b, c, d, and e. These variables can be of any type.
						`"hello".unpacket(0, a, b, c, d, e);`

						This method does not unpack multi-byte types like 4-byte integers or floats.


					upper:

						Converts a string to uppercase, returning a new string.

						string string::upper();

					Returns:

						string: the specified string in uppercase.

					Example:

						```NVGT
						void main() {
							string text = input_box("Text", "Enter the text to convert to uppercase.");
							if (text.empty()) {
								alert("Info", "You typed a blank string.");
								exit(1);
							}
							alert("Your string uppercased is", text.upper());
						}



					upper_this:

						Converts a string to uppercase, modifying the calling string.

						string& string::upper();

					Returns:

						string&: a reference to the specified string in uppercase.

					Example:

						```NVGT
						void main() {
							string text = input_box("Text", "Enter the text to convert to uppercase.");
							if (text.empty()) {
								alert("Info", "You typed a blank string.");
								exit(1);
							}
							text = text.upper_this();
							alert("Your string uppercased is", text);
						}



		Streams:
			datastreams:
				Streams, also known as datastreams, are one of the primary ways of moving around and manipulating data in nvgt. With their convenient function calls, low memory footprint when dealing with large datasets and with their ability to connect to one another to create a chain of mutations on any data, datastreams are probably the most convenient method for data manipulation in NVGT that exist.

				A datastream could be a file on disk, a file downloading from/uploading to the internet, a simple incapsulated string, or some sort of a manipulator such as an encoder, decryptor or compression stream.

				Datastreams can roughly be split into 3 categories, or if you want to be really specific 2.5.
				* sources: These streams are things like file objects, internet downloads, or really anything that either inputs new data into the application or outputs data from it. Thus, they can read, write or both depending on the properties of the stream.
				* readers: These streams only support read/input operations, an example may be a decoder. Usually these attach to another stream, read data from the connected stream, and mutate that data as the stream is read from.
				* writers: The polar opposite of readers, these streams usually connect to another stream E. a file object opened in writing mode where data gets mutated (usually encoded) before being written to the connected stream.

				Occasionally, you may see a reader referred to as a decoder, and a writer referred to as an encoder.

				Particularly when considering reader and writer streams that manipulate data, you can typically chain any number of streams of the same type together to cause a sequence of data mutations to be performed in one function call. For example you could connect an inflating_reader to a hex_decoder that is in turn connected to a file object in read mode. From that point, calling any of the read functions associated with the inflating stream would automatically first read from the file, hex decode it, and decompress/inflate it as needed. Inversely, you could connect a deflating_writer to a hex encoder which is connected to a file object in write mode, causing any data written to the inflating stream to be compressed, hex encoded, and finally written to the file.

				There are a few readers and writers that, instead of manipulating data in any way as they pass through, give you details about that data. You can see counting_reader and counting_writer as examples of this, these streams count the number of lines and characters that are read or written through them.

				Lets put this together with a little demonstration. Say you have a compressed and hex encoded file with many lines in it, and you'd like to read the file while determining the line count as the file is read. You could execute the following, for example:
				counting_reader f(inflating_reader(hex_decoder(file("test.txt", "rb"))));
				string result;
				while(f.good()) {
					result += f.read(100);
					alert("test", "currently on line " + f.lines + " on character position " + f.pos);
				}
				f.close();
				alert("test", "The file contains a total of " + f.lines + " and after decoding, contains the following text: \n" + result);

				More datastreams could be added to the engine at any time.

				The available datastreams are listed below in this subsection of the documentation.

				All streams are derived from the datastream class, so you will want to have a look at the documentation for that class first before seeing what extras are provided by the datastream you want to work with.


			datastream:

				The base class for all datastreams, this stream can read and write to an internal string buffer maintained by the class if constructed directly. Any other datastream can also be cast to the "datastream" type to facilitate passing datastreams of any type throughout the application, and thus any child datastreams such as encoders, file sources or any others will contain the functions listed here as they are derived from the datastream class.

				1. datastream();

				2. datastream(string initial_data = "", string encoding = "", datastream_byte_order byteorder = STREAM_BYTE_ORDER_NATIVE);

			Arguments (2):

				* string initial_data = "": The initial contents of this stream derived from a string.

				* string encoding = "": The text encoding used to read or write strings to this stream (see remarks), this argument appears in all non-empty child datastream constructors.

				* datastream_byte_order byteorder = STREAM_BYTE_ORDER_NATIVE: The byte order to read or write binary data to this stream using (see remarks), this argument appears in all non-empty child datastream constructors.

			Remarks:

				If this class is directly instantiated, the effect is basically that the string class gets wrapped with streaming functions. The default datastream class will create an internal string object, and any data passed into the initial_data parameter will be copied to that internal string. Then, the read/write functions on the stream will do their work on the internal string object held by the datastream class, which can be read or retrieved in full at any time. Internally, this wraps the std::stringstream class in c++.

				It must be noted, however, that this is the parent class for all other datastreams in nvgt. This means that any child datastream such as the file class can be cast to a generic datastream handle. In this case, the read/write functions for the casted handle will perform the function of the child stream which has been casted instead of on an internal string. This is the same for any other parent/child class relationship in programming, but it was mentioned here to avoid any confusion between the default datastream implementation and a datastream handle casted from a different stream.

				The encoding argument, present in nearly all child datastream  constructors, controls what encoding if any strings should be converted to from UTF8 as they are written to the stream with the write_string() function or << operator while the binary property on the stream is set to true, as well as what encoding to convert from when reading them with read_string() or the >> operator. If set to an empty string (the default), strings are left in UTF8 when writing, and already expected to be in UTF8 in a stream when reading from it.

				The byteorder argument, again present in nearly all child datastream constructors, controls what endianness is used when reading/writing binary data from/to a stream, that is the read_int/write_float/similar functions when the binary property on the stream is set to true. When a value takes more than one byte, the endianness or byte order controls whether the bytes of that value are read/written from left to right or right to left, or in proper terms whether the most significant byte of the value should be written first. The values that can be accepted here are:

				* STREAM_BYTE_ORDER_NATIVE (default): The byte order used by the system the script is running on.

				* STREAM_BYTE_ORDER_BIG_ENDIAN: The most significant byte is read/written first.

				* STREAM_BYTE_ORDER_NETWORK: Same as STREAM_BYTE_ORDER_BIG_ENDIAN, provided because this is indeed a very common name for the big endian byte order as it is typically used for data transmission.

				* STREAM_BYTE_ORDER_LITTLE_ENDIAN: The most significant byte is read/written last.

				Usually, you can leave the byteorder value at the default for most streams. However in some situations where you are transmitting binary data between systems running on different architectures, it may be better to set the transmitting and receiving streams of such an application to a common byte order that is not system native.

				Though this was mentioned above, it's worth reiterating once more that all other stream types contain all of the functions listed in this base datastream class unless otherwise noted, and thus are not documented multiple times in child classes.

				If an initial_data argument is provided when constructing a datastream, the stream will be set at the beginning, ready to read the initial data rather than at the end. If you wish to append more data to the datastream after it is constructed, you should call the seek_end() method on it first.

			Example:

				```NVGT
				void main() {
					datastream ds1("This is a demonstration.");
					alert("example", ds1.read()); // Will display "This is a demonstration."
					datastream ds2;
					ds2.write("Hello there, ");
					ds2.write("good bye.");
					ds2.seek(0);
					alert("example", ds2.read()); // Will display "Hello there, good bye."
					// The following shows how this datastream can be used as an area to store encoded data.
					datastream encoded;
					hex_encoder h(encoded); // We attach the encoded datastream to a hex encoder.
					h.write("I am a hex string"); // "I am a hex string" in hex is written to the datastream object called encoded.
					h.close();
					encoded.seek(0);
					alert("example", hex_decoder(encoded).read()); // We attach a hex_decoder to the encoded datastream and read from it, thus this will display "I am a hex string".
				}



				Methods:
					close:

						Close a datastream, freeing any associated resources and leaving the datastream in an inactive state.

						bool close(bool close_connected = false);

					arguments:

						* bool close_connected = false: Whether to also close any streams that are connected to this one (see remarks).

					Returns:

						bool: true on success, false on failure such as if the stream is already closed.

					Remarks:

						Generally it is best to call this function when you are done working with any stream, particularly when dealing with files, encoders, or any stream that is opened in write mode. In reality there are a few streams, such as the default datastream class, where it is OK to not call the close method.

						The reason that it is sometimes OK to not call the close function on a stream when you are done with it is because this function is automatically called in the internal destructor for any datastream, meaning when any datastream object is destroyed, it's close method will be called. For some streams like the default datastream class which is just wrapping a string, this is fine (you don't close a string, after all), but in many cases you will want to close a stream at a certain time before it is destroyed. For example closing an encoding stream may write a few final characters to a connected file stream as it is closing, meaning that the close method on the encoder must be called prior to the close function on the stream connected to it, something which you may not have control of when objects are getting destructed.

						The close_connected argument controls whether to also close any streams that are connected to the stream that close is being called on. For the default datastream class that wraps a string or for any other datastream that doesn't connect to another one, this argument has no effect.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.close();
							alert("example", ds.active); // Will display false, as the stream is no longer active.
							alert("example", ds.write("this is a test")); // Will return 0 instead of a positive number of bytes written as the stream is not opened.
							ds.open();
							alert("example", ds.write("this is a test")); // Now returns 14, as expected.
							// Calling the close method a final time is not required for this example because it uses an instance of the default datastream class. It will be taken care of when the script exits.
						}



					close_all:

						Close a datastream as well as any that are connected to it.

						bool close_all();

					Returns:

						bool: true if the stream and any connected to it could be closed, false otherwise.

					Remarks:

						This is exactly the same thing as calling the close method with the close_connected boolean argument set to true. On the default datastream class it will have no effect besides freeing the internal string, however you can look at the example below to see a case where there is an effect by calling close_all().

					Example:

						```NVGT
						void main() {
							datastream ds;
							hex_encoder h(ds);
							h.write("hi");
							h.close_all(); // Will also close the datastream called ds because it is connected to the hex encoder.
							alert("example", ds.active); // Will display false, indicating that calling h.close_all() also caused ds.close() to implicitly be called.
						}



					get_pos:

						Retrieve the offset of a datastream.

						int get_pos();

					Arguments:

						None.

					Returns:

						int: Returns the byte offset of the stream.

					Example:

						```NVGT
						void main() {
							datastream ds;
							if (!ds.open("this is a demonstration")) {
								alert("Oh no", "Maybe this could happen for a file but it should never happen for a basic datastream!");
								exit();
							}
							ds.seek(10);
							alert("Position", ds.get_pos());
						}



					open:

						Open a datastream, connecting it to a source or another stream depending on type.

						bool open(...);

					Arguments:

						Same as in the constructor for the datastream you wish to open (see remarks).

					Returns:

						bool: Returns true if the stream could be opened, false otherwise.

					Remarks:

						Most datastreams can be closed and then reopened, or can be created in an uninitialized state meaning they must be opened to begin with.

						All datastreams can be constructed in an already initialized/opened state, and the constructors that allow this for each stream will contain the exact same arguments as that streams associated open() function, and each streams constructor topics is where such arguments are documented.

						For example, the true signature for the open function on the default datastream class is bool open(string initial_data = "", string encoding = "", datastream_byteorder byteorder = STREAM_BYTE_ORDER_NATIVE) while the signature of the open function for the file datastream is bool open(string filename, string mode, string encoding = "", datastream_byte_order byteorder = STREAM_BYTE_ORDER_NATIVE).

						As mentioned in the top level datastreams topic, the encoding and byteorder arguments are present in each stream and thus will not be redocumented for each one.

						For all streams, this function will call the associated close() function prior to opening the requested resource if the stream is already active at this time. If it turns out that users do not desire this behaviour, it may be made optional or could get removed in the future.

					Example:

						```NVGT
						void main() {
							datastream ds;
							if (!ds.open("this is a demonstration")) {
								alert("Oh no", "Maybe this could happen for a file but it should never happen for a basic datastream!");
								exit();
							}
							alert("example", ds.read()); // Will display "this is a demonstration".
						}



					read:

						Read raw bytes from a stream.

						string read(uint amount = 0);

					Arguments:

						* uint amount = 0: The number of bytes to read, or 0 to read the entire stream. Only applicable for strings.

					Returns:

						string: The data that was read from the stream or an empty string on failure.

					Remarks:

						If the length of the string returned by this function is less than the number of bytes requested in the amount argument, either the end of the stream was reached or there was an error. You can check what happened by evaluating the fail or eof properties on the stream.

						This is the lowest level method of reading from a stream. All other reading functions either read a certain datatype or until a certain condition is met.

						This should not be confused with the read_string function.

						Typically it is only useful to call this function if the good property on the stream is true, therefor you can query the stream's good property in a loop to see if you should continue reading data if you are trying to perform some sort of buffered reading of a stream.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read(6)); // Will display Hello followed by a space.
							alert("example", ds.read()); // Will display there, I am a string wrapped in a sstream!
						}



					read_double:

						Read raw bytes from a stream into a double.

						double read_double();

					Arguments:

						None.

					Returns:

						double: The data that was read, or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read_double()); // Will return the first 8 bytes of the string as a double.
						}



					read_float:

						Read raw bytes from a stream into a float.

						float read_float();

					Arguments:

						None.

					Returns:

						float: The data that was read, or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read_float()); // Will return the first 4 bytes of the string as a float.
						}



					read_int:

						Read raw bytes from a stream into a 32-bit integer.

						int read_int();

					Arguments:

						None.

					Returns:

						int: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read_int()); // Will display the first 4 bytes of the string as an integer.
						}



					read_int16:

						Read raw bytes from a stream into a 16-bit integer.

						int16 read_int16();

					Arguments:

						None.

					Returns:

						int16: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read_int16()); // Will display the first two bytes of the string as an integer.
						}



					read_int64:

						Read raw bytes from a stream into a 64-bit integer.

						int64 read_int64();

					Arguments:

						None.

					Returns:

						int64: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read_int64()); // Will display the first eight bytes of the string as an integer.
						}



					read_int8:

						Read raw bytes from a stream into an 8-bit integer.

						int8 read_int8();

					Arguments:

						None.

					Returns:

						int8: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read_int8()); // Will display the first byte of the string as an integer (72).
						}



					read_string:

						Read a string from a stream.

						string read_string();

					Arguments:

						None.

					Returns:

						string: The data that was read from the stream or an empty string on failure.

					Remarks:

						This method behaves differently depending on the binary flag:

						* If binary is true, it expects the number of bytes to read to be prepended to the string as a uint8.

						* If binary is false, it will read a single word from the string using any whitespace as its boundary.

						Typically it is only useful to call this function if the good property on the stream is true, therefor you can query the stream's good property in a loop to see if you should continue reading data if you are trying to perform some sort of buffered reading of a stream.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there,\nI am a string wrapped in a sstream!");
							ds.binary=false;
							alert("example", ds.read_string()); // Returns "Hello".
							alert("example", ds.read_string()); // Returns "there,".
						}



					read_uint:

						Read raw bytes from a stream into a 32-bit unsigned integer.

						uint read_uint();

					Arguments:

						None.

					Returns:

						uint: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read_uint()); // Will display the first four bytes of the string as an unsigned integer.
						}



					read_uint16:

						Read raw bytes from a stream into a 16-bit unsigned integer.

						uint16 read_uint16();

					Arguments:

						None.

					Returns:

						uint16: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read_uint16()); // Will display the first two bytes of the string as an unsigned integer.
						}



					read_uint64:

						Read raw bytes from a stream into a 64-bit unsigned integer.

						uint64 read_uint64();

					Arguments:

						None.

					Returns:

						uint64: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read_uint64()); // Will display the first eight bytes of the string as an unsigned integer.
						}



					read_uint8:

						Read raw bytes from a stream into an 8-bit unsigned integer.

						uint8 read_uint8();

					Arguments:

						None.

					Returns:

						uint8: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds("Hello there, I am a string wrapped in a sstream!");
							alert("example", ds.read_uint8()); // Will display the first byte of the string as an unsigned integer (72).
						}



					seek:

						Seek to an offset in the datastream.

						bool seek(int offset);

					Arguments:

						offset: The byte offset you wish to seek to.

					Returns:

						bool: Returns true if the stream could seek, false otherwise.

					Remarks:

						Note that most datastreams do not support seeking as this would result in needing to store more data in memory than we are comfortable with, however, file objects and raw datastreams that read from a string do support seeking. Other than these stream types, you should assume that seek operations are not supported unless otherwise documented.

					Example:

						```NVGT
						void main() {
							datastream ds;
							if (!ds.open("this is a demonstration")) {
								alert("Oh no", "Maybe this could happen for a file but it should never happen for a basic datastream!");
								exit();
							}
								ds.seek(10);
						alert("example", ds.read()); // Will display "demonstration".
						}



					write:

						Write raw bytes to a stream.

						uint write(string content);

					Arguments:

						* content: The content that is to be written.

					Returns:

						uint: the number of bytes written.

					Remarks:

						If needed, you can use the returned value to verify whether the data you intended to write to the stream was written successfully by checking whether the return value of this function matches the length of the data you passed to it.

					Example:

						```NVGT
						void main() {
							// This time we'll use a file which is another type of datastream to show how the functions in this datastream class work on it's children.
							file f("test.txt", "wb");
							string data = "This is a test file";
							bool success = f.write(data) == data.length();
							f.close();
							alert("Information", "The file with the data has been " + (success? "successfully" : "unsuccessfully") + " written");
						}



					write_double:

						Write the value of a double into a stream.

						uint write_double(double value);

					Arguments:

						* value: The value that is to be written.

					Returns:

						uint: the number of bytes written.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.write_double(5.5);
						}



					write_float:

						Write the value of a float into a stream.

						uint write_float(float content);

					Arguments:

						* content: The content that is to be written.

					Returns:

						uint: the number of bytes written.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.write_float(5.5);
						}



					write_int:

						Write the value of a 32-bit integer into a stream.

						uint write_int(int content);

					Arguments:

						* content: The content that is to be written.

					Returns:

						uint: the number of bytes written.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.write_int(4);
						}



					write_int16:

						Write the value of a 16-bit integer into a stream.

						uint write_int16(int16 content);

					Arguments:

						* content: The content that is to be written.

					Returns:

						uint: the number of bytes written.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.write_int16(4);
						}



					write_int64:

						Write the value of a 64-bit integer into a stream.

						uint write_int64(int64 content);

					Arguments:

						* content: The content that is to be written.

					Returns:

						uint: the number of bytes written.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.write_int64(4);
						}



					write_int8:

						Write the value of an 8-bit integer into a stream.

						uint write_int8(int8 content);

					Arguments:

						* content: The content that is to be written.

					Returns:

						uint: the number of bytes written.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.write_int8(4);
						}



					write_string:

						Write a string into a stream.

						uint write_string(string content);

					Arguments:

						* content: The content that is to be written.

					Returns:

						uint: the number of bytes written.

					Remarks:

						This method behaves differently depending on the binary flag:

						* If binary is true, it prepends the number of bytes to write as a uint8.

						* If binary is false, it will merely write the entire string.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.binary=true;
							ds.write_string("Hello");
							alert("Test", ds.read());
						}



					write_uint:

						Write the value of a 32-bit unsigned integer into a stream.

						uint write_uint(uint content);

					Arguments:

						None.

					Returns:

						uint: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.write_uint(8);
						}



					write_uint16:

						Write the value of a 16-bit unsigned integer into a stream.

						uint write_uint16(uint16 content);

					Arguments:

						None.

					Returns:

						uint16: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.write_uint16(1);
						}



					write_uint64:

						Write the value of a 64-bit unsigned integer into a stream.

						uint write_uint64(uint64 content);

					Arguments:

						None.

					Returns:

						uint64: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.write_uint64(50);
						}



					write_uint8:

						Write the value of an 8-bit unsigned integer into a stream.

						uint write_uint8(uint8 content);

					Arguments:

						None.

					Returns:

						uint8: The data that was read from the stream or 0 on failure.

					Example:

						```NVGT
						void main() {
							datastream ds;
							ds.write_uint8(12);
						}



				Properties:
					active:
						This property will be true if a stream is opened and ready for use, false otherwise.

						`const bool active;`



					available:

						This property returns the number of bytes immediately available to be read from a stream.

						const int available;

					Remarks:

						This property may not be implemented in all streams, for example decoders. If it is unavailable, it will return 0.

					Example:

						```NVGT
						void main() {
							datastream ds("example");
							alert("example", ds.available); // Displays 7.
							ds.read(2);
							alert("example", ds.available); // Now shows 5, as 2 of the 7 bytes have been read.
						}



					bad:

						This property will be true if the previous function failed with a fatal error.

						const bool bad;

					Remarks:

						This is good for checking the success or failure of the last operation. It differs from fail in that this is generally set in unrecoverable errors (such as device failure or unreadable data).

					Example:

						```NVGT
						void main() {
							datastream ds("Hi");
							ds.read();
							alert("example", ds.bad); // If this returns true, something is seriously wrong.
						}



					binary:
						Set this property to true to work with binary data, or false to work with text. Defaults to true.

						`bool binary;`



					eof:

						This property will be true on a stream when it has no more data to read.

						const bool eof;

					Example:

						```NVGT
						void main() {
							datastream ds("hello");
							alert("example", ds.eof); // Will show false because we always start at the beginning of a default datastream, thus there is data to read.
							ds.read();
							alert("example", ds.eof); // Will now show true because any further .read calls will fail as we have reached the end of the stream.
							ds.seek(3);
							alert("example", ds.eof); // Will again show false because we are no longer at the end of the stream.
						}



					fail:

						This property will be true if the previous function failed.

						const bool fail;

					Remarks:

						This is good for checking the success or failure of the last operation. It differs from bad in that this is generally set in recoverable errors (EOF, type mismatch and so on).

					Example:

						```NVGT
						void main() {
							datastream ds;
							alert("example", ds.fail); // Will display false because we haven't done anything yet.
							ds.read(); // Error, nothing there.
							alert("example", ds.fail); // Will now display true because there was an error reading.
						}



					good:

						This property will be true when a datastream is ready to read or write, such as when the active property is true and the end of file has not been reached.

						const bool good;

					Remarks:

						This property is specifically true so long as the stream is opened and the eof, fail, and bad properties all return false.

					Example:

						```NVGT
						void main() {
							datastream ds("hello");
							alert("example", ds.good); // Will display true because there is data to read.
							ds.read();
							alert("example", ds.good); // Will now display false because the end of file has been reached, ds.eof is true now.
						}



			file:
				The file datastream is used to read and write files stored on the hard disk.

				1. `file();`
				2. `file(const string path, const string mode);`

			Arguments (1):
				* const string path: the filename to open.

			Arguments (2):
				* const string path: the filename to open.
				* const string mode: the mode to open as.

			Remarks:
				Usually when the file object is first created, it will not be active, that is, it will not be associated with a file on disk. To activate it, use the following methods:

				* Call the open function.
				* use the second constructor.

				Please note that both methods require the filename that is to be associated and the mode to open, with the only difference being that it is harder to tell whether the file was opened successfully if you use the constructor rather than the open method. Using the second constructor makes it 1 line shorter. The possible open modes will not be documented in this remarks, you can see it in `file::open` method.

				Remember that as with all datastreams, all methods in the base datastream class will work on the file class unless noted otherwise and thus will not be redocumented here.


				Methods:
					open:

						This method will open a file for reading or writing.

						bool file::open(string filename, string open_mode);

					Arguments:

						* string filename: the name of the file to open. This can be either an absolute or relative path.

						* string open_mode: the mode to open, see remarks.

					Returns:

						bool: true on success, false on failure.

					Remarks:

						While on some operating systems (mostly windows) both the slash(`/`), and the backslash(`\`) can be used to specify the filename, it is very strongly recommended to use the / character for greatest cross platform compatibility.

						The following is a list of valid open modes.

						* a: append.

						* w: write.

						* r: read.

						* r+: read and write.

						For backwards compatibility with code that used a version of the file object from when there was a difference between text and binary file open modes, a b character is also accepted (for example rb) to indicate binary. The current stream implementation ignores this character other than to gracefully accept it rather than complaining that it is an invalid mode. You can use other text encoding APIs such as string_recode and the line_converting_reader if you really need to try recreating something similar to the old behavior.

						The file will be created if the file does not exist if opened in either write or append mode. When opened in read mode, the file must exist in order to be successful.

					Example:

						```NVGT
						void main() {
							file f;
							f.open("test.txt", "wb");
							f.write("This is a test");
							f.close();
							alert("Information", "The file has been written");
						}



				Properties:
					size:

						Determine the size (in bytes) of the file that is currently associated with this stream.

						const uint64 size;

					Remarks:

						This property will be 0 if no file is associated with this stream, in which case you can use the datastream::active property on it to check the difference between a 0 byte file and an eronious result.

						When files are opened in write mode, there could be periods where the written data has not yet been flushed to disk in which case this property may have a value which is a little bit behind. Experiments seem to indicate that this rarely if never happens, however it's worth putting the note here just encase anyone runs into it.

						Note that a file_get_size() function exists in the engine which is usually better than this property unless you need to do more with a file than just get it's size.

					Example:

						```NVGT
						void main() {
							file f("size.nvgt", "rb");
							if (!f.active) {
								alert("oops", "couldn't load the file");
								return;
							}
							alert("size.nvgt is", f.size + "b");
							f.close();
						}



		Audio:
			Classes:
				sound:
					Methods:
						close:

							Closes an opened sound, making the sound object available to be reloaded with a new one.

							bool sound::close();

						Returns:

							bool: true if the sound could be closed, false otherwise for example if there is no sound opened.

						Example:

							```NVGT
							void main() {
								sound s;
								s.load("C:/windows/media/ding.wav");
								alert("example", s.close()); // Will display true since a sound was previously opened.
								alert("example", s.close()); // Will now display false since the previous operation freed the sound object of any attached sound.
							}



						load:
							Loads a sound file with the specified settings.

							1. `bool sound::load(string filename, pack@ soundpack = null, bool allow_preloads = true);`
							2. `bool sound::load(sound_close_callback@ close_cb, sound_length_callback@ length_cb, sound_read_callback@ read_cb, sound_seek_callback@ seek_cb, string data, string preload_filename = "");`

						Arguments (1):
							* string filename: the name/path to the file that is to be loaded.
							* pack@ soundpack = null: a handle to the pack object to load this sound from, if any.
							* bool allow_preloads = true: whether or not the sound system should preload sounds into memory on game load.

						Arguments (2):
							* sound_close_callback@ close_cb: the close callback to use with the sound (see remarks).
							* sound_length_callback@ length_cb: the length callback to use with the sound (see remarks).
							* sound_read_callback@ read_cb: the read callback to use with the sound (see remarks).
							* sound_seek_callback@ seek_cb: the seek callback to use with the sound (see remarks).
							* string data: the audio data of the sound.
							* string preload_filename = "": the name of the file to be preloaded (if any).

						Returns:
							bool: true if the sound was successfully loaded, false otherwise.

						Remarks:
							The syntax for the sound_close_callback is:
							> void sound_close_callback(string user_data);

							The syntax for the sound_length_callback is:
							> uint sound_length_callback(string user_data);

							The syntax for the sound_read_callback is:
							> int sound_read_callback(string &out buffer, uint length, string user_data);

							The syntax for the sound_seek_callback is:
							> bool sound_seek_callback(uint offset, string user_data);


						load_url:

							Load the specified URL.

							bool sound::load_url(string url);

						Arguments:

							* string url: the URL to load (should be the direct link to a sound or stream).

						Returns:

							Bool: True if the URL was successfully loaded; false otherwise.

						Example:

							```NVGT
							void main() {
								sound s;
								bool success = s.load_url("https://example.com/my_sound.mp3");
								alert("Result", "The sound was " + (success == false? "not " : "") + "successfully loaded.");
							}



						pause:
							Pauses the sound.

							`bool sound::pause();`

						Returns:
							bool: true if the sound was paused, false otherwise.


						play:
							Starts playing the sound.

							`bool sound::play();`

						Returns:
							bool: true if the sound was able to start playing, false otherwise.


						play_looped:
							Starts playing the sound repeatedly.

							`bool sound::play_looped();`

						Returns:
							bool: true if the sound was able to start playing, false otherwise.


						play_wait:
							Starts playing the sound, blocking the calling thread until it's finished.

							`bool sound::play_wait();`

						Returns:
							bool: true if the sound has successfully loaded and finished playing, false otherwise.


						seek:
							Seeks to a particular point in the sound.

							`bool sound::seek(float ms);`

						Arguments:
							* float ms: the time (in MS) to seek to.

						Returns:
							bool: true if the seeking was successful, false otherwise.


						set_position:
							Sets the sound's position in 3d space.

							`bool sound::set_position(float listener_x, float listener_y, float listener_z, float sound_x, float sound_y, float sound_z, float rotation, float pan_step, float volume_step);`

						Arguments:
							* float listener_x: the x position of the listener.
							* float listener_y: the y position of the listener.
							* float listener_z: the z position of the listener.
							* float sound_x: the x position of the sound.
							* float sound_y: the y position of the sound.
							* float sound_z: the z position of the sound.
							* float rotation: the rotation of the listener (in radians).
							* float pan_step: the pan step (e.g. how extreme the panning is).
							* float volume_step: the volume step (very similar to pan_step but for volume).

						Returns:
							bool: true if the position was successfully set, false otherwise.


						stop:
							Stops the sound, if currently playing.

							`bool sound::stop();`

						Returns:
							bool: true if the sound was successfully stopped, false otherwise.


					Properties:
						active:
							Determine if the sound has successfully been loaded or not.

							`bool sound::active;`


						length:
							Get the length of a sound (in milliseconds).

							`float sound::length;`


						loaded_filename:
							Obtain the path of sound file that has been loaded.

							`string sound::loaded_filename;`


						paused:
							Determine if the sound is paused.

							`bool sound::paused;`


						playing:
							Determine if the sound is currently playing.

							`bool sound::playing;`


						sliding:
							Determine if a sound is sliding or not.

							`sound::sliding;`


			Functions:
				get_sound_input_devices:

					Return an array of strings listing all input sound devices on the system.

					string[]@ get_sound_input_devices();

				Returns:

					string[]@: A handle to an array listing all available input devices.

				Remarks:

					After listing devices with this function, you can set the sound_input_device global engine property to whatever index in the returned array contains the name of the device you want to set.

					This list will always include a no sound device. If you don't want the user to be able to select it, feel free to not allow them to move their cursor to that item or not add it to your menus etc, but remember to pay attention to the index difference this may create when setting the sound_output_device property, namely just remember that setting sound_output_device = 0 will set the system to no sound, and setting the device to 1 will always be the first real device in the system, or the default device at least on windows.

				Example:

					```NVGT
					void main() {
						string[]@ devices = get_sound_input_devices();
						devices.remove_at(0); // Don't allow the user to select no sound.
						alert("devices", "The following devices are installed on your system: " + join(devices, ", "));
					}



				get_sound_output_devices:

					Return an array of strings listing all output sound devices on the system.

					string[]@ get_sound_output_devices();

				Returns:

					string[]@: A handle to an array listing all available output devices.

				Remarks:

					After listing devices with this function, you can set the sound_output_device global engine property to whatever index in the returned array contains the name of the device you want to set.

					This list will always include a no sound device. If you don't want the user to be able to select it, feel free to not allow them to move their cursor to that item or not add it to your menus etc, but remember to pay attention to the index difference this may create when setting the sound_output_device property, namely just remember that setting sound_output_device = 0 will set the system to no sound, and setting the device to 1 will always be the first real device in the system, or the default device at least on windows.

				Example:

					```NVGT
					void main() {
						string[]@ devices = get_sound_output_devices();
						devices.remove_at(0); // Don't allow the user to select no sound.
						alert("devices", "The following devices are installed on your system: " + join(devices, ", "));
					}



			Global Properties:
				sound_default_mixer:
					Represents the default mixer object for all your sounds to use.

					`mixer@ sound_default_mixer;`


				sound_default_pack:
					The default value passed to the second argument of sound::load, in other words, a handle to an open pack object that all future sounds will load from unless otherwise specified individually.

					`pack@ sound_default_pack = null;`


				sound_global_hrtf:
					Controls weather to use steam audio's functionality. If this property is set to false, the sound_environment class will be quite useless and sounds will use very basic panning.

					`bool sound_global_hrtf;`

					Changes take nearly instant effect from the time this property is modified.


		Concurrency:
			This section contains the documentation for all mechanisms revolving around several things running at once, usually involving threads and their related synchronization facilities.

			A warning that delving into this section will expose you to some rather low level concepts, the misapplication of which could result in your program crashing or acting oddly without the usual helpful error information provided by NVGT.

			The highest level and most easily invoked method for multithreading in nvgt is the async template class, allowing you to call any script or system function on another thread and retrieve its return value later.



			Classes:
				async:

					A template class that allows one to very easily call any function in nvgt on another thread and fetch that function's return value after the call has complete.

					1. async\<T\>();

					2. async\<T\>(const ?&in function, const ?&in arg1, const ?&in arg2, ...);

				Template arguments:

					* T: The return type of the function being called.

				Arguments (2):

					* const ?&in function: The function that should be called, passing anything that is not a function will result in an exception.

					* const ?&in arg1: The function's first argument, you do not need to pass more arguments than needed.

					* const ?&in arg2: The function's second argument, passing invalid or mismatched datatypes as any of these arguments will cause an exception.

					* ... Up to 15 arguments.

				Remarks:

					Generally speaking, this is the most convenient method of applying multithreading to your nvgt application. While lower level methods of dealing with threads require the programmer to create functions with specific signatures that usually don't allow for return values, this class allows you to quite transparently call any function in the application from a function in your script to the alert box built into NVGT on another thread while still maintaining the return value of such a function for later retrieval.

					For ease of use, the constructor of this class actually calls the function provided and passes the given arguments to it. Therefor, the standard use case for this class is to create an instance of it, then to first call instance.wait() or instance.try_wait(ms) before retrieving the instance.value variable when the function's return value is needed, after either instance.wait() returns or after instance.try_wait() returns true. Using the instance.complete/instance.failed properties with your own waiting logic is also perfectly acceptable and sometimes recommended.

					The one drawback that makes this class look a little less pretty is that if you wish to call functions that are part of a class, you must create funcdefs for the signature of the function you want to call, then wrap the object's function in an instance of the funcdef. For example if a class contained a function bool load(string filename), you must declare funcdef void my_void_funcdef(string); and if you then had an instance of such a class called my_object, you must initialize the async object like async\<bool\>(my_void_funcdef(my_object.load), "my_file.txt"); however it is a relatively minor drawback.

					Be aware that this class can throw exceptions if you do not pass arguments to the function correctly, this is because the function call happens completely at runtime rather than being prepared at compilation time like the rest of the script.

					Internally, this function is registered with Angelscript multiple times with an expanding number of arguments, meaning that it is OK to pass only the number of arguments you need to a function you want to call even though this function's signature seems to indicate that the dynamically typed arguments here don't have a default value.

				Example:

					```NVGT
					void main() {
						show_window("example");
						async<int>@ answer = async<int>(question, "exit program", "Do you want to exit the program?");
						// Now that the user sees the dialog, lets play a sound.
						sound s;
						s.load("c:/windows/media/ding.wav");
						s.play_looped();
						// We will move the sound around the stereo field just to drive home the point that we can execute our own code while the dialog is showing.
						int pan = 0;
						while(true) {
							wait(25); // A slower value than average for the sake of the pan effect, will make window a bit unresponsive though (use a timer).
							pan += random(-2, 2);
							if (pan < -100) pan = -100;
							else if (pan > 100) pan = 100;
							s.pan = pan;
							if (answer.complete) { // The async function has successfully finished and a return value is available.
								if (answer.value == 1) exit();
								else @answer = async<int>(question, "exit program", "Now do you want to exit the program?");
							}
						}
					}



					Methods:
						try_wait:

							Wait a given number of milliseconds for an async call to complete.

							bool try_wait(uint milliseconds);

						Arguments:

							* uint milliseconds: The amount of time to wait, may be 0 to not wait at all.

						Returns:

							bool: True if the async call has finished, or false if it is still running after the given number of milliseconds has expired.

						Remarks:

							If you try waiting for 50ms but the async call finishes in 10ms, the try_wait call will only block for 10 out of the 50ms requested. In short, any blocking try_wait call is canceled prematurely with the try_wait function returning true if the call finishes while try_wait is in the middle of executing.

							If the function has already finished executing or if this async object is not connected to a function (thus nothing to wait for), try_wait() will immediately return true without waiting or blocking at all regardless of arguments.

							Similar to the async::wait function, you should be careful using this method in the same thread that show_window was called or if you do, make sure to call it with tiny durations and interspurse standard nvgt wait(5) or similar calls in your loops so that window messages and events continue to be handled.

						Example:

							```NVGT
							void main() {
								// Pan a sound around while an alert box shows, no window.
								async<int> call(alert, "hi", "press ok to exit");
								sound s;
								s.load("c:/windows/media/ding.wav");
								s.play_looped();
								while (!call.try_wait(40)) { // We'll use the blocking provided by try_wait for our timer.
									s.pan += random(-3, 3);
								}
							}



						wait:

							Wait for the async function to finish executing.

							void wait();

						Remarks:

							This is the simplest way to wait for the execution of an async function to complete, however it does not allow you to execute any of your own code while doing so. The execution of your program or the thread you called the wait function on will be completely blocked until the async function returns. This also means that window events cannot be received while the wait function is executing, so you should be very careful about using this from an environment with a game window especially on the thread that created it.

							For more control such as the ability to execute your own code during the wait, check out the try_wait() function.

							If the async object this function is called on is executed with a default constructor and thus has no function call in progress, this function will return immediately as there is nothing to wait for.

						Example:

							```NVGT
							void main() {
								async<string> result(input_box, "name", "please enter your name");
								result.wait(); // Input_box creates a dialog on it's own thread so the remark about windowing doesn't apply in this situation.
								alert("test", "hello " + result.value); // May not focus automatically because from a different thread than the input box.
							}



					Properties:
						complete:

							Immediately shows whether an async function call has completed, meaning that either a result is available or that there has been an error.

							const bool complete;

						Remarks:

							This is the best and prettiest looking method of checking whether an async call has completed or not without blocking. You can also call the try_wait() function with an argument of 0 for a similar effect, though this looks nicer and takes any uninitialized state of the object into account.

							Usually this will be used when a loop has some other sort of waiting logic, such as the global nvgt wait() function we are all familiar with.

						Example:

							```NVGT
							void main() {
								sound s; // Press space to play even while alert box is opened.
								s.load("c:/windows/media/ding.wav");
								async<int> call(alert, "test", "press ok to exit"); // May need to alt+tab to it, window is shown after.
								show_window("example"); // Shown after alert because otherwise alert will be child of the window.
								while(!call.complete) {
									wait(5);
									if (key_pressed(KEY_SPACE)) {
										s.stop();
										s.play();
									}
								}
							}



						exception:

							A string describing any exception that took place during an async function call.

							const string exception;

						Remarks:

							If there has been no error or if the async object has no attached function call, a blank string will be returned.

						Example:

							```NVGT
							funcdef void return_void_taking_uint(uint);
							void main() {
								string[] my_empty_array;
								async<void> result(return_void_taking_uint(my_empty_array.remove_at), 0); // We are calling my_empty_array.remove_at(0) on another thread, sure to cause an exception because the array is empty.
								result.wait();
								alert("test", result.exception);
							}



						failed:

							Returns true if an async function call has thrown an exception.

							const bool failed;

						Remarks:

							This is a shorthand version of executing the expression (async.try_wait(0) and async.exception != ""), provided for syntactical ease.

						Example:

							```NVGT
							string throw_exception_randomly() {
								if (random_bool(50)) throw("oh no!");
								return "yo yo";
							}
							void main() {
								async<string> result(throw_exception_randomly);
								result.wait();
								if (result.failed) alert("oops", result.exception);
								else alert("success", result.value);
							}



						value:

							Contains the return value of a successful async function call.

							const T& value;

						Remarks:

							Remember that this class is a template type, so T will adapt to whatever type was being used when the async object was constructed.

							If a value is not yet available, the wait function will be internally called upon first access to this property, meaning that accessing this property could cause your program to block until data is available. You are meant to call the wait/try_wait functions or check the complete property first before accessing this.

							If an async object is initialized with the default constructor meaning it has no function attached, for example async\<string\> result; accessing result.value will throw an exception because no data will ever be available in this context.

						Example:

							```NVGT
							void main() {
								// Lets demonstrate the edge cases mentioned above as most examples in the documentation for this class show off this property being used normally.
								async<string> result1(input_box, "type text", "enter a value");
								alert("test", result1.value); // The main thread will block until result1.value is available. Be careful!
								async<sound@> result2; // This is not connected to a function, maybe the object could be reassigned to a result later.
								sound@ s = result2.value; // Will throw an exception!
							}



				atomic_flag:
					An `atomic_flag` is a fundamental synchronization primitive that represents the simplest form of an atomic boolean flag that supports atomic test-and-set and clear operations. The `atomic_flag` type is specifically designed to guarantee atomicity without the need for locks, ensuring that operations on the flag are performed as indivisible actions even in the presence of concurrent threads. Unlike `atomic_bool`, `atomic_flag` does not provide load or store operations.

					```nvgt
					atomic_flag();

				Remarks:
					Unlike all other atomic types, `atomic_flag` is guaranteed to be lock-free.


					methods:
						clear:
							Atomically changes the state of an `atomic_flag` to clear (false). If order is one of MEMORY_ORDER_ACQUIRE or MEMORY_ORDER_ACQ_REL, the behavior is undefined.

							```nvgt
							void clear(memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `order`: the memory synchronization ordering.


						notify_all:
							Unblocks all threads blocked in atomic waiting operations (i.e., `wait()`) on this `atomic_flag`, if there are any; otherwise does nothing.

							```nvgt
							void notify_all();

						Remarks:
							This form of change detection is often more efficient than pure spinlocks or polling and should be preferred whenever possible.



						notify_all:
							Unblocks at least one thread blocked in atomic waiting operations (i.e., `wait()`) on this `atomic_flag`, if there is one; otherwise does nothing.

							```nvgt
							void notify_one();

						Remarks:
							This form of change detection is often more efficient than pure spinlocks or polling and should be preferred whenever possible.


						test:
							Atomically reads the value of this `atomic_flag` and returns it. The behavior is undefined if the memory order is `MEMORY_ORDER_RELEASE` or `MEMORY_ORDER_ACQ_REL`.

							```nvgt
							bool test(memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `order`: the memory synchronization ordering.

						Returns:
							The value atomically read.


						test_and_set:
							Atomically changes the value of this `atomic_flag` to set (`true`) and returns it's prior value.

							```nvgt
							bool test_and_set(memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `order`: the atomic synchronization order.

						Returns:
							The prior value of this `atomic_flag`.


						wait:
							Atomically waits until the value of this `atomic_flag` has changed. If order is either `MEMORY_ORDER_RELEASE` or `MEMORY_ORDER_ACQ_REL`, the behavior is undefined.

							```nvgt
							void wait(bool old, memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `old`: The old (current) value of this `atomic_flag` as of the time of this call. This function will wait until this `atomic_flag` no longer contains this value.
							* `order`: memory order constraints to enforce.

						Remarks:
							This function is guaranteed to return only when the value has changed, even if the underlying implementation unblocks spuriously.


				atomic_T:

					The base class for all atomic types that NVGT has to offer.

				Remarks:

					An atomic type SHALL be defined as a data type for which operations must be performed atomically, ensuring that modifications are indivisible and uninterrupted within a concurrent execution environment. An atomic operation MUST either fully succeed or completely fail, with no possibility of intermediate states or partial completion.

					When one thread writes to an atomic object and another thread reads from the same atomic object, the behavior MUST be well-defined and SHALL NOT result in a data race.

					Operations on atomic objects MAY establish inter-thread synchronization and MUST order non-atomic memory operations as specified by the `memory_order` parameter. Memory effects MUST be propagated and observed according to the constraints of the specified memory ordering.

					Within this documentation, the `atomic_T` class is a placeholder class for any atomic type. Specifically, `T` may be any primitive type except void, but MAY NOT be any complex type such as string. For example, `atomic_bool` is an actual class, where `bool` replaces `T`. Additionally, an `atomic_flag` class exists which does not offer load or store operations but is the most efficient implementation of boolean-based atomic objects, and has separate documentation from all other atomic types.

					Please note: atomic floating-point types are not yet implemented, though they will be coming in a future release. However, until then, attempts to instantiate an atomic floating-point type will behave as though the class in question did not exist. This notice will be removed once atomic floating-point types have been implemented.

					Methods:
						compare_exchange_strong:
							Atomically compares the value representation of this atomic object with that of `expected`. If both are bitwise-equal, performs an atomic read-modify-write operation on this atomic object with `desired` (that is, replaces the current value of this atomic object with `desired`); otherwise, performs an atomic load of this atomic object and places it's actual value into `expected`. If failure is either `MEMORY_ORDER_RELEASE` or `MEMORY_ORDER_ACQ_REL`, the behavior is undefined. 

							1: `bool compare_exchange_strong(T& expected, T desired, memory_order success, memory_order failure);`
							2: `bool compare_exchange_strong(T& expected, T desired, memory_order order = MEMORY_ORDER_SEQ_CST);`

						Parameters (1):
							* `T& expected`: reference to the value expected to be found in this atomic object.
							* `T desired`: the value that SHALL replace the one in this atomic object if and only if it is bitwise-equal to `expected`.
							* `memory_order success`: the memory synchronization ordering that SHALL be used for the read-modify-write operation if the comparison succeeds.
							* `memory_order failure`: the memory synchronization ordering that SHALL be used for the load operation if the comparison fails.
							* `memory_order order`: the memory synchronization order that SHALL be used for both the read-modify-write operation and the load operation depending on whether the comparison succeeds or fails.

						Returns:
							`true` if the atomic value was successfully changed, false otherwise.

						Remarks:
							This function is available on all atomic types.

							Within the above function signatures, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							The operations of comparison and copying SHALL be executed in a bitwise manner. Consequently, no invocation of a constructor, assignment operator, or similar function SHALL occur, nor SHALL any comparison operators be utilized during these operations.

							In contrast to the `compare_exchange_weak` function, this function SHALL NOT spuriously fail.

							In scenarios where the use of the `compare_exchange_weak` function would necessitate iteration, whereas this function would not, the latter SHALL be considered preferable. Exceptions to this preference exist in cases where the object representation of type T might encompass trap bits or offer multiple representations for the same value, such as floating-point NaNs. Under these circumstances, `compare_exchange_weak` generally proves effective, as it tends to rapidly converge upon a stable object representation.


						compare_exchange_weak:
							Atomically compares the value representation of this atomic object with that of `expected`. If both are bitwise-equal, performs an atomic read-modify-write operation on this atomic object with `desired` (that is, replaces the current value of this atomic object with `desired`); otherwise, performs an atomic load of this atomic object and places it's actual value into `expected`. If failure is either `MEMORY_ORDER_RELEASE` or `MEMORY_ORDER_ACQ_REL`, the behavior is undefined. 

							```nvgt
							bool compare_exchange_weak(T& expected, T desired, memory_order success, memory_order failure);
							bool compare_exchange_weak(T& expected, T desired, memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `T& expected`: reference to the value expected to be found in this atomic object.
							* `T desired`: the value that SHALL replace the one in this atomic object if and only if it is bitwise-equal to `expected`.
							* `memory_order success`: the memory synchronization ordering that SHALL be used for the read-modify-write operation if the comparison succeeds.
							* `memory_order failure`: the memory synchronization ordering that SHALL be used for the load operation if the comparison fails.
							* `memory_order order`: the memory synchronization order that SHALL be used for both the read-modify-write operation and the load operation depending on whether the comparison succeeds or fails.

						Returns:
							bool: `true` if the atomic value was successfully changed, false otherwise.

						Remarks:
							This function is available on all atomic types.

							Within the above function signatures, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							The operations of comparison and copying SHALL be executed in a bitwise manner. Consequently, no invocation of a constructor, assignment operator, or similar function SHALL occur, nor SHALL any comparison operators be utilized during these operations.

							In contrast to the `compare_exchange_strong` function, this function MAY fail spuriously. Specifically, even in instances where the value contained within this atomic object is equivalent to the expected value, the function is permitted to act as if the values are not equal. This characteristic allows the function to provide enhanced performance on certain platforms, particularly when employed within iterative loops.

							In scenarios where the use of this function would necessitate iteration, whereas `compare_exchange_strong` would not, the latter SHALL be considered preferable. Exceptions to this preference exist in cases where the object representation of type T might encompass trap bits or offer multiple representations for the same value, such as floating-point NaNs. Under these circumstances, this function generally proves effective, as it tends to rapidly converge upon a stable object representation.


						exchange:
							Atomically replaces the value of this object with `desired` in such a way that the operation is a read-modify-write operation, then returns the prior value of this object. Memory is affected according to `order`.

							```nvgt
							T exchange(T desired, memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `T desired`: the value to exchange with the prior value.
							* `memory_order order`: the memory ordering constraints to enforce.

						Returns:
							T: The prior value held within this atomic object before this function was called.

						Remarks:
							This function is available on all atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.


						fetch_add:
							Atomically replaces the current value with the result of arithmetic addition of the value and `arg`. That is, it performs atomic post-increment. The operation is a read-modify-write operation. Memory is affected according to the value of `order`.

							```nvgt
							T fetch_add(T arg, memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `T arg`: the value to add to this atomic object.
							* `memory_order order`: which memory order SHALL govern this operation.

						Returns:
							T: The prior value of this atomic object.

						Remarks:
							This function is only available on integral and floating-point atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.


						fetch_and:
							Atomically replaces the current value with the result of bitwise ANDing the value of this atomic object and `arg`. The operation is a read-modify-write operation. Memory is affected according to the value of `order`.

							```nvgt
							T fetch_and(T arg, memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `T arg`: the right-hand side of the bitwise AND operation.
							* `memory_order order`: which memory order SHALL govern this operation.

						Returns:
							The prior value of this atomic object.

						Remarks:
							This function is only available on integral atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.


						fetch_or:
							Atomically replaces the current value with the result of bitwise ORing the value of this atomic object and `arg`. The operation is a read-modify-write operation. Memory is affected according to the value of `order`.

							```nvgt
							T fetch_or(T arg, memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `T arg`: the right-hand side of the bitwise OR operation.
							* `memory_order order`: which memory order SHALL govern this operation.

						Returns:
							T: The prior value of this atomic object.

						Remarks:
							This function is only available on integral atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.


						fetch_sub:
							Atomically replaces the current value with the result of arithmetic subtraction of the value and `arg`. That is, it performs atomic post-decrement. The operation is a read-modify-write operation. Memory is affected according to the value of `order`.

							```nvgt
							T fetch_sub(T arg, memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `T arg`: the value to subtract from this atomic object.
							* `memory_order order`: which memory order SHALL govern this operation.

						Returns:
							T: The prior value of this atomic object.

						Remarks:
							This function is only available on integral and floating-point atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.


						fetch_xor:
							Atomically replaces the current value with the result of bitwise XORing the value of this atomic object and `arg`. The operation is a read-modify-write operation. Memory is affected according to the value of `order`.

							```nvgt
							T fetch_xor(T arg, memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `T arg`: the right-hand side of the bitwise XOR operation.
							* `memory_order order`: which memory order SHALL govern this operation.

						Returns:
							T: The prior value of this atomic object.

						Remarks:
							This function is only available on integral atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.


						is_lock_free:
							Checks whether the atomic operations on all objects of this type are lock-free.

							```nvgt
							bool is_lock_free();

						Returns:
							bool: true if the atomic operations on the objects of this type are lock-free, `false` otherwise.

						Remarks:
							This function is available on all atomic types.

							All atomic types, with the exception of `atomic_flag`, MAY be implemented utilizing mutexes or alternative locking mechanisms as opposed to employing lock-free atomic instructions provided by the CPU. This allows for the implementation flexibility where atomicity is achieved through synchronization primitives rather than hardware-based atomic instructions.

							Atomic types MAY exhibit lock-free behavior under certain conditions. For instance, SHOULD a particular architecture support naturally atomic operations exclusively for aligned memory accesses, then any misaligned instances of the same atomic type MAY necessitate the use of locks to ensure atomicity.

							While it is recommended, it is NOT a mandatory requirement that lock-free atomic operations be address-free. Address-free operations are those that are suitable for inter-process communication via shared memory, facilitating seamless data exchange without reliance on specific memory addresses.

						load:
							Atomically loads and returns the current value of the atomic variable. Memory is affected according to the value of `order`.  If order is either `MEMORY_ORDER_RELEASE` or `MEMORY_ORDER_ACQ_REL`, the behavior is undefined.

							```nvgt
							T load(memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `memory_order order`: which memory order to enforce when performing this operation.

						Returns:
							T: The value of this atomic object.

						Remarks:
							This function is available on all atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							This operation is identical to using the IMPLICIT CONVERSION OPERATOR, except that it allows for the specification of a memory order when performing this operation. When using the IMPLICIT CONVERSION OPERATOR, the memory order SHALL be `MEMORY_ORDER_SEQ_CST`.

						notify_all:
							Unblocks all threads blocked in atomic waiting operations (i.e., `wait()`) on this atomic object if there are any; otherwise does nothing.

							```nvgt
							void notify_all();

						Remarks:
							This function is available on all atomic types.

							This form of change detection is often more efficient than pure spinlocks or polling and should be preferred whenever possible.



						notify_one:
							Unblocks at least one thread blocked in atomic waiting operations (i.e., `wait()`) on this  atomic object if there is one; otherwise does nothing.

							```nvgt
							void notify_one();

						Remarks:
							This function is available on all atomic types.

							This form of change detection is often more efficient than pure spinlocks or polling and should be preferred whenever possible.


						store:
							Atomically replaces the current value with `desired`. Memory is affected according to the value of `order`.  If order is either MEMORY_ORDER_ACQUIRE or MEMORY_ORDER_ACQ_REL, the behavior is undefined. 

							```nvgt
							void store(T desired, memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `T desired`: the value that should be stored into this atomic object.
							* `memory_order order`: which memory ordering constraints should be enforced during this operation.

						Remarks:
							This function is available on all atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							This operation is identical to using the assignment operator, except that it allows for the specification of a memory order when performing this operation. When using the assignment operator, the memory order SHALL be `MEMORY_ORDER_SEQ_CST`.

						wait:
							Atomically waits until the value of this atomic object has changed. If order is either `MEMORY_ORDER_RELEASE` or `MEMORY_ORDER_ACQ_REL`, the behavior is undefined.

							```nvgt
							void wait(T old, memory_order order = MEMORY_ORDER_SEQ_CST);

						Parameters:
							* `T old`: The old (current) value of this atomic object as of the time of this call. This function will wait until this atomic object no longer contains this value.
							* `memory_order order`: memory order constraints to enforce.

						Remarks:
							This function is available on all atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							This function is guaranteed to return only when the value has changed, even if the underlying implementation unblocks spuriously.


					Operators:
						opAddAssign:
							Atomically replaces the current value with the result of computation involving the previous value and `arg`. The operation is a read-modify-write operation. Specifically, performs atomic addition. Equivalent to `return fetch_add(arg) + arg;`.

							```nvgt
							T opAddAssign( T arg );

						Returns:
							T: The resulting value (that is, the result of applying the corresponding binary operator to the value immediately preceding the effects of the corresponding member function in the modification order of this atomic object).

						Remarks:
							This operator is only available on integral and floating-point atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							Unlike most compound assignment operators, the compound assignment operators for atomic types do not return a reference to their left-hand arguments. They return a copy of the stored value instead. 


						opAndAssign:
							Atomically replaces the current value with the result of computation involving the previous value and `arg`. The operation is a read-modify-write operation. Specifically, performs atomic bitwise AND. Equivalent to `return fetch_and(arg) & arg;`.

							```nvgt
							T opAndAssign(T arg);

						Returns:
							The resulting value of this computation.

						Remarks:
							This operator is only available on integral atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							Unlike most compound assignment operators, the compound assignment operators for atomic types do not return a reference to their left-hand arguments. They return a copy of the stored value instead. 


						opAssign:
							Atomically assigns desired to the atomic variable. Equivalent to `store(desired)`.

							```nvgt
							T opAssign(T desired);

						Remarks:
							This operator is available on all atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							Unlike most assignment operators, the assignment operators for atomic types do not return a reference to their left-hand arguments. They return a copy of the stored value instead.


						opImplConv:
							Atomically loads and returns the current value of the atomic variable. Equivalent to `load()`.

							```nvgt
							T opImplConv();

						Returns:
							T: The current value of the atomic variable. 

						Remarks:
							This operator is available on all atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

						opOrAssign:
							Atomically replaces the current value with the result of computation involving the previous value and `arg`. The operation is a read-modify-write operation. Specifically, performs atomic bitwise OR. Equivalent to `return fetch_or(arg) | arg;`.

							```nvgt
							T opOrAssign(T arg);

						Returns:
							T: The resulting value of this computation.

						Remarks:
							This operator is only available on integral atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							Unlike most compound assignment operators, the compound assignment operators for atomic types do not return a reference to their left-hand arguments. They return a copy of the stored value instead. 


						opPostDec:
							Atomically decrements the current value. The operation is a read-modify-write operation. Specifically, performs atomic post-decrement. Equivalent to `return fetch_add(1);`.

							```nvgt
							T opPostDec(int arg);

						Remarks:
							This operator is only available on integral atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							Unlike most post-decrement operators, the post-decrement operators for atomic types do not return a reference to the modified object. They return a copy of the stored value instead. 


						opPostInc:
							Atomically increments the current value. The operation is a read-modify-write operation. Specifically, performs atomic post-increment. Equivalent to `return fetch_add(1);`.

							```nvgt
							T opPostInc(int arg);

						Remarks:
							This operator is only available on integral atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							Unlike most post-increment operators, the post-increment operators for atomic types do not return a reference to the modified object. They return a copy of the stored value instead. 


						opPreDec:
							Atomically decrements the current value. The operation is a read-modify-write operation. Specifically, performs atomic pre-decrement. Equivalent to `return fetch_sub(1) - 1;`.

							```nvgt
							T opPreDec();

						Remarks:
							This operator is only available on integral atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							Unlike most pre-decrement operators, the pre-decrement operators for atomic types do not return a reference to the modified object. They return a copy of the stored value instead. 


						opPreInc:
							Atomically increments the current value. The operation is a read-modify-write operation. Specifically, performs atomic pre-increment. Equivalent to `return fetch_add(1) + 1;`.

							```nvgt
							T opPreInc();

						Remarks:
							This operator is only available on integral atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							Unlike most pre-increment operators, the pre-increment operators for atomic types do not return a reference to the modified object. They return a copy of the stored value instead. 


						opSubAssign:
							Atomically replaces the current value with the result of computation involving the previous value and `arg`. The operation is a read-modify-write operation. Specifically, performs atomic subtraction. Equivalent to `return fetch_sub(arg) + arg;`.

							```nvgt
							T opSubAssign( T arg );

						Returns:
							T: The resulting value (that is, the result of applying the corresponding binary operator to the value immediately preceding the effects of the corresponding member function in the modification order of this atomic object).

						Remarks:
							This operator is only available on integral and floating-point atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							Unlike most compound assignment operators, the compound assignment operators for atomic types do not return a reference to their left-hand arguments. They return a copy of the stored value instead. 


						opXorAssign:
							Atomically replaces the current value with the result of computation involving the previous value and `arg`. The operation is a read-modify-write operation. Specifically, performs atomic bitwise XOR. Equivalent to `return fetch_xor(arg) ^ arg;`.

							```nvgt
							T opXorAssign(T arg);

						Returns:
							T: The resulting value of this computation.

						Remarks:
							This operator is only available on integral atomic types.

							Within the above function signature, `T` is used as a placeholder for the actual type. For example, if this object is an `atomic_int`, then `T` SHALL be `int`.

							Unlike most compound assignment operators, the compound assignment operators for atomic types do not return a reference to their left-hand arguments. They return a copy of the stored value instead.


					Properties:
						is_always_lock_free:
							This property equals true if this atomic type is always lock-free and false if it is never or sometimes lock-free. 

							```nvgt
							bool get_is_always_lock_free() property;

						Remarks:
							This property is available on all atomic types. This property is read-only.


				mutex:

					This multithreading primitive allows for any number of threads to safely access a shared object while avoiding illegal concurrent access.

					mutex();

				Remarks:

					It is not safe for 2 threads to try accessing the same portion of memory at the same time, that especially becomes true when at least one of the threads performs any kind of updates to the memory. One of the common and widely used solutions to this problem is basically to create a lock condition where, if one thread starts modifying a bit of shared data, another thread which wants to access or also modify that data must first wait for the initial accessing thread to complete that data modification before continuing.

					The mutex class is one such method of implementing this mechanism. On the surface it is a simple structure that can be in a locked or unlocked state. If thread A calls the lock method on a mutex object which is already locked from thread B, thread A will block execution until thread B calls the unlock method of the mutex object which can then be successfully locked by thread A.

					This is the most basic mutex class, though there are other derivatives. Most mutex classes have the ability to try locking a mutex for up to a given timeout in milliseconds before returning false if a lock could not be acquired within the given timeout.

					Mutexes or most other threading primitives are not connected by default to any sort of data. You can use one mutex for 10 variables if you want and if it makes sense in your context.

					Remember, you are beginning to enter lower level and much more dangerous programming territory when dealing with multiple threads. If you accidentally forget to unlock a mutex when you are done with it for example, any thread that next tries locking that mutex will at the least malfunction but is much more likely to just hang your app forever. Be careful!

				Example:

					```NVGT
					// globals
					mutex ding_mutex; // To control exclusive access to a sound object.
					sound snd;
					thread_event exit_program(THREAD_EVENT_MANUAL_RESET);
					// This function plays a ding sound every 250 milliseconds, the sleeping is handled by trying to wait for the thread exit event.
					void dinging_thread() {
						while(!exit_program.try_wait(250)) {
							ding_mutex.lock();
							snd.seek(0);
							snd.play();
							ding_mutex.unlock();
						}
					}
					void main() {
						snd.load("C:/Windows/media/chord.wav");
						show_window("mutex example");
						// Start our function on another thread.
						async<void>(dinging_thread);
						while (!key_pressed(KEY_ESCAPE)) {
							wait(5);
							if (key_pressed(KEY_L)) {
								// This demonstrates that if we lock the mutex on the main thread, the ding sound pauses for a second while it's own ding_mutex.lock() call waits for this thread to unlock the mutex.
								ding_mutex.lock();
								wait(1000);
								ding_mutex.unlock();
							}
							// A more real-world use case might be allowing the sounds volume to safely change.
							if (key_pressed(KEY_UP)) {
								mutex_lock exclusive(ding_mutex); // A simpler way to lock a mutex, the mutex_lock variable called exclusive gets deleted as soon as this if statement scope exits and the mutex is then unlocked.
								snd.volume += 1;
							} else if (key_pressed(KEY_DOWN)) {
								mutex_lock exclusive(ding_mutex);
								snd.volume -= 1; // We can be sure that these volume changes will not take place while the sound is being accessed on another thread.
							}
						}
						exit_program.set(); // Make sure the dinging_thread knows it's time to shut down.
					}



					methods:
						lock:
							Locks the mutex, waiting indefinitely or for a given duration if necessary for the operation to succeed.

							1. `void lock();`
							2. `void lock(uint milliseconds);`

						Arguments (2):
							* uint milliseconds: How long to wait for a lock to succeed before throwing an exception

						Remarks:
							With the mutex class in particular, it is safe to call the lock function on the same thread multiple times in a row so long as it is matched by the same number of unlock calls. This may not be the case for other types of mutexes.

							Beware that if you use the version of the lock function that takes a timeout argument, an exception will be thrown if the timeout expires without a lock acquiring. The version of the function taking 0 arguments waits forever for a lock to succeed.


						try_lock:

							Attempts to lock the mutex.

							1. bool try_lock();

							2. bool try_lock(uint milliseconds);

						Arguments (2):

							* uint milliseconds: The amount of time to wait for a lock to acquire.

						Returns:

							bool: true if a lock was acquired, false otherwise.

						Remarks:

							This is the method to use if you want to try locking a mutex without blocking your programs execution. The 0 argument version of this function will return false immediately if the mutex is already locked on another thread, while the version of the method taking a milliseconds value will  wait for the timeout to expire before returning false on failure.

							Note that several operating systems do not implement this functionality natively or in an easily accessible manner, such that the version of this function that takes a timeout might enter what amounts to a `while(!try_lock()) wait(0);` loop while attempting to acquire a lock. If the operating system supports such a timeout functionality natively which will be a lot faster such as those using pthreads, that will be used instead.

						Example:

							```NVGT
							// Press space to perform a calculation, and press enter to see the result only so long as the calculation is not in progress.
							thread_event keep_calculating;
							bool exit_program = false; // NVGT will wrap atomic flags later, for now we'll piggyback off the keep_calculating event.
							mutex calculation_mutex;
							int calculation;
							void do_calculations() {
								while (true) {
									keep_calculating.wait();
									if (exit_program) return;
									// Lets increase the calculation variable for a bit.
									screen_reader_speak("calculating...", true);
									timer t;
									calculation_mutex.lock();
									while(t.elapsed < 1000) calculation++;
									calculation_mutex.unlock();
									screen_reader_speak("complete", true);
								}
							}
							void main() {
								async<void>(do_calculations); // Spin up our thread.
								show_window("try_lock example");
								while (!key_pressed(KEY_ESCAPE)) {
									wait(5);
									if (key_pressed(KEY_SPACE)) keep_calculating.set();
									if (key_pressed(KEY_RETURN)) {
										if (calculation_mutex.try_lock()) {
											screen_reader_speak(calculation, true);
											calculation_mutex.unlock();
										} else screen_reader_speak("calculation in progress", true);
									}
								}
								exit_program = true;
								keep_calculating.set();
							}



						unlock:
							Unlocks the mutex.

							`void unlock();`

						Remarks:
							This function does nothing if the mutex is already unlocked.


					mutex_lock:
						Lock a mutex until the current execution scope exits.

						1. `mutex_lock(mutex@ mutex_to_lock);`
						2. `mutex_lock(mutex@ mutex_to_lock, uint milliseconds);`

					Arguments (2):
							* uint milliseconds: The amount of time to wait for a lock to acquire before throwing an exception.

					Remarks:
						Often it can become tedious or sometimes even unsafe to keep performing mutex.lock and mutex.unlock calls when dealing with mutex, and this object exists to make that task a bit easier to manage.

						This class takes advantage of the sure rule that, unless handles become involved, an object created within a code scope will be automatically destroyed as soon as that scope exits.

						For example:

						if (true) {
							string s = "hello";
						} // the s string is destroyed when the program reaches this brace.

						In this case, the constructor of the mutex_lock class will automatically call the lock method on whatever mutex you pass to it, while the destructor calls the unlock method. Thus, you can lock a mutex starting at the line of code a mutex_lock object was created, which will automatically unlock whenever the scope it was created in is exited.

						The class contains a single method, void unlock(), which allows you to unlock the mutex prematurely before the scope actually exits.

						There is a very good reason for using this class sometimes as opposed to calling mutex.lock directly. Consider:

						my_mutex.lock();
						throw("Oh no, this code is broken!");
						my_mutex.unlock();

						In this case, mutex.unlock() will never get called because an exception got thrown meaning that the rest of the code down to whatever handles the exception won't execute! However we can do this:

						mutex_lock exclusive(my_mutex);
						throw("Oh no, this code is broken!");
						exclusive.unlock()

						The mutex_lock object will be destroyed as part of the exception handler unwinding the stack, because the handler already knows to destroy any objects it encounters along the way. However any code that handles exceptions certainly does not know it should specifically call the unlock method of a mutex, thus you could introduce a deadlock in your code if you lock a mutex and then run code that throws an exception without using this mutex_lock class.

						A final hint, it is actually possible to create scopes in Angelscript and in many other languages as well without any preceding conditions, so you do not need an if statement or a while loop to use this class.

						int var1 = 2;
						string var2 = "hi";
						{
							string var3 = "catch me if you can...";
						}
						string var4 = "Hey, where'd var3 go!";


				thread_event:

					This concurrency primative implements a method by which one or more threads can safely wait on another thread for the event to become signaled.

					thread_event(thread_event_type type = THREAD_EVENT_AUTO_RESET);

				Arguments:

					thread_event_type type: The type of handling to perform when a waiting thread detects that the event has been set (see remarks).

				Remarks:

					This primative can be used to make one thread inexpensively sleep until another thread wakes it up, or can more simply be used for some threads to monitor when a condition has been completed on another thread in a safe manner.

					It is not a good idea to just use standard booleans in multithreaded situations where you want one thread to know when another thread has completed it's work because threads could be reading from the boolean as it is being written to by a worker thread, causing an undefined result. The thread_event object is one way around this issue, as it can signal when a standard boolean is safe to access or sometimes avoid their usage entirely.

					Quite simply one thread can call the set() method on an event, and if another thread has previously called the wait() method on the same event, that wait method will return thus resuming the first thread. If the event was created with the THREAD_EVENT_AUTO_RESET type, the event will also be set to unsignaled as the waiting thread wakes up. Otherwise, the reset() method must be called manually.

				Example:

					```NVGT
					// Globals
					thread_event program_event;
					bool exit_program;
					string message;
					void message_thread() {
						while (true) {
							program_event.wait();
							if (exit_program) return;
							screen_reader_speak(message, true);
						}
					}
					void main() {
						show_window("press space to speak on another thread");
						async<void>(message_thread);
						while (!key_pressed(KEY_ESCAPE)) {
							wait(5);
							if (key_repeating(KEY_SPACE)) {
								message = "the date and time is " + calendar().format(DATE_TIME_FORMAT_RFC1123);
								program_event.set();
							}
						}
						exit_program = true;
						program_event.set();
					}



					Methods:
						reset:
							Resets an event to an unsignaled state.

							`void reset();`

						Remarks:
							This method is usually only needed if THREAD_EVENT_MANUAL_RESET was specified when the event was created, however you could also cancel the signaling of an event if another thread hasn't detected the signal yet.

							See the main event and mutex chapters for examples.


						set:
							Causes an event to become signaled, waking up any threads waiting for this condition.

							`void set();`

						Remarks:
							All existing examples that use events must call this method, so it will not be demonstrated in this chapter.


						try_wait:

							Wait a given number of milliseconds for an event to become signaled.

							bool try_wait(uint milliseconds);

						Arguments:

							* uint milliseconds: The amount of time to wait, may be 0 to not wait at all.

						Returns:

							bool: True if another thread has signaled the event, or false if the event is still unsignaled after the given number of milliseconds has elapsed.

						Remarks:

							The semantics of this function are exactly the same as mutex::try_wait(), accept that this function waits on an event instead of a mutex.

						Example:

							```NVGT
							thread_event g_wait;
							void test_thread() {
								screen_reader_speak("started", true);
								while (!g_wait.try_wait(1000)) screen_reader_speak("waiting...", true);
								screen_reader_speak("ah!", true);
								g_wait.set();
							}
							void main() {
								async<void>(test_thread);
								wait(3500);
								g_wait.set();
								g_wait.wait(); // So we'll know the other thread has finished speaking it's final message.
							}



						wait:
							waits for the event to become signaled, blocking indefinitely or for a given duration if required.

							1. `void wait();`
							2. `void wait(uint milliseconds);`

						Arguments (2):
							* uint milliseconds: How long to wait for the event to become signaled before throwing an exception

						Remarks:
							Beware that if you use the version of the wait function that takes a timeout argument, an exception will be thrown if the timeout expires without the event having become signaled. The version of the function taking 0 arguments waits forever for the event's set() method to be called.


			Enums:
				`memory_order`:
					Specifies the constraints on the ordering and visibility of memory operations (reads and writes) in concurrent programming, determining how operations on shared memory are observed across different threads.

					| Constant | Description |
					| --- | --- |
					| `MEMORY_ORDER_RELAXED` | An atomic operation with `MEMORY_ORDER_RELAXED` has no synchronization or ordering constraints beyond those imposed by the atomicity of the operation itself. It guarantees that the operation on the atomic variable is atomic and modifications to that variable are visible in some modification order consistent across threads, but it does not impose any inter-thread ordering constraints or create happens-before relationships. |
					| `MEMORY_ORDER_ACQUIRE` | An atomic operation with `MEMORY_ORDER_ACQUIRE` on an atomic variable synchronizes with a release operation on the same variable from another thread. It ensures that all memory writes in other threads that release the same atomic variable become visible in the current thread before any subsequent memory operations (following the acquire operation) are performed. This establishes a happens-before relationship from the release to the acquire. |
					| `MEMORY_ORDER_RELEASE` | An atomic operation with `MEMORY_ORDER_RELEASE` ensures that all preceding memory operations in the current thread are completed before the release operation is performed. It makes the effects of these prior operations visible to other threads that perform an acquire operation on the same atomic variable. The release operation synchronizes with an acquire operation on the same variable, establishing a happens-before relationship. |
					| `MEMORY_ORDER_ACQ_REL` | An atomic operation with `MEMORY_ORDER_ACQ_REL` combines both acquire and release semantics. For operations that modify the atomic variable (read-modify-write operations), it ensures that all preceding memory operations in the current thread are completed before the operation (release semantics), and that all subsequent memory operations are not started until after the operation (acquire semantics). This enforces that the operation synchronizes with other acquire or release operations on the same variable, establishing happens-before relationships in both directions. |
					| `MEMORY_ORDER_SEQ_CST` | An atomic operation with `MEMORY_ORDER_SEQ_CST` (sequential consistency) provides the strongest ordering guarantees. It ensures that all sequentially consistent operations appear to occur in a single total order that is consistent with the program order in all threads. This total order is interleaved with the program order such that each read sees the last write to that variable according to this order. It combines the effects of acquire and release semantics and enforces a global order of operations, establishing a strict happens-before relationship. |


				thread_event_type:
					These are the possible event types that can be used to successfully construct thread_event objects.
					* THREAD_EVENT_MANUAL_RESET: Many threads can wait on this event because the `thread_event::reset()` method must be manually called.
					* THREAD_EVENT_AUTO_RESET: Only one thread can wait on this event because it will be automatically reset as soon as the waiting thread detects that the event has become signaled.


				thread_priority:
					It is possible to set a thread's priority to the following values in NVGT:
					* THREAD_PRIORITY_LOWEST
					* THREAD_PRIORITY_LOW
					* THREAD_PRIORITY_NORMAL
					* THREAD_PRIORITY_HIGH
					* THREAD_PRIORITY_HIGHEST


			Functions:
				thread_current_id:

					Determine the operating system specific ID of the currently executing thread.

					uint thread_current_id();

				Returns:

					uint: The ID of the thread that this function was called on.

				Example:

					```NVGT
					void main() {
						alert("Current thread ID is", thread_current_id());
					}



				thread_sleep:
					Sleeps the thread it was called from, but can be interrupted.

					`bool thread_sleep(uint milliseconds);`

				Arguments:
					* uint milliseconds: the number of milliseconds to sleep the thread for.

				Returns:
					bool: true if the thread slept for the full duration, false if it was interrupted by `thread.wake_up`.

				Remarks:
					This function should only be called in the context of threads created within your script.


				thread_yield:
					Yields CPU to other threads.
					void thread_yield();

				Remarks:
					This is a little bit like executing `wait(0);` accept that it doesn't pull the window, and there is absolutely no sleeping. It will temporarily yield code execution to the next scheduled thread that is of the same or higher priority as the one that called this function.


		Data Manipulation:
			Classes:
				json_array:
					Methods:
						add:

							Add a value to the end of a JSON array.

							void json_array::add(var@ value);

						Arguments:

							* var@ value: a handle to the value to be set.

						Example:

							```NVGT
							void main() {
								json_array@ arr = parse_json("[]");
								for (uint i = 0; i < 5; i++)
									arr.add(random(1, 100));
								alert("Filled array", arr.stringify());
							}



						clear:

							Clears the JSON array completely.

							void json_array::clear();

						Example:

							```NVGT
							void main() {
								string data = "[1, 2, 3, 4, 5]";
								json_array@ arr = parse_json(data);
								alert("Before being cleared", arr.stringify());
								arr.clear();
								alert("After being cleared", arr.stringify());
							}



						is_array:

							Determine if the value at the given index is a JSON array or not.

							bool json_array::is_array(uint index);

						Arguments:

							* uint index: the position of the item.

						Returns:

							bool: true if the value at the specified index is a JSON array, false otherwise.

						Example:

							```NVGT
							void main() {
								json_array@ arr = parse_json("[[1, 2], [3, 4]]");
								alert("Info", arr.is_array(1) ? "It is an array": "It is not an array");
							}



						is_null:

							Determine if the value at the given index is null.

							bool json_array::is_null(uint index);

						Arguments:

							* uint index: the position of the item.

						Returns:

							bool: true if the value at the specified index is null, false otherwise.

						Example:

							```NVGT
							void main() {
								json_array@ arr = parse_json("[1, 2, null, 4]");
								alert("Info", "Index 1 " + (arr.is_null(1) ? " is null": " is not null"));
								alert("Info", "Index 2 " + (arr.is_null(2) ? " is null": " is not null"));
							}



						is_object:

							Determine if the value at the given index is a JSON object.

							bool json_array::is_object(uint index);

						Arguments:

							* uint index: the position of the item.

						Returns:

							bool: true if the value at the specified index is a JSON object, false otherwise.

						Example:

							```NVGT
							void main() {
								json_array@ arr = parse_json("""[{}, {}, "test", {}]""");
								alert("Info", "Position 0 " + (arr.is_object(0) ? "is an object" : "is not an object"));
								alert("Info", "Position 2 " + (arr.is_object(2) ? "is an object": "is not an object"));
							}



						remove:

							Remove a value from the JSON array given an index.

							void json_array::remove(uint index);

						Arguments:

							* uint index: the index of the value to remove.

						Example:

							```NVGT
							void main() {
								string data = """[47, 4, 584, 43, 8483]""";
								json_array@ arr = parse_json(data);
								alert("Initial", arr.stringify());
								arr.remove(2);
								alert("After removal", arr.stringify());
							}



						size:

							Return the size of the JSON array.

							uint json_array::size();

						Example:

							```NVGT
							void main() {
								string data = "[";
								for (int i = 0; i < random(10, 20); i++)
									data += i + ", ";
								// Strip off the trailing comma and space character, as they'll make the JSON throw a syntax error.
								data.erase(data.length() - 1);
								data.erase(data.length() - 1);
								data += "]";
								clipboard_set_text(data);
								json_array@ arr = parse_json(data);
								alert("Info", "The JSON array contains " + arr.size() + " items");
							}



						stringify:

							Return the JSON array as a string.

							string json_array::stringify(int indent = 0, int step = -1);

						Arguments:

							* int indent = 0: specifies how many spaces to use to indent the JSON. If 0, no indentation or formatting is performed.

							* int step = -1: the outwards indentation step, for example setting it to two would cause your outdents to only be two spaces. -1 (the default) keeps it how it is, normally you don't need to use this parameter.

						Returns:

							string: the JSON array as a string.

						Example:

							```NVGT
							void main() {
								string data = """["Apple", "Banana", "Orange", "Lemon"]""";
								json_array@ arr = parse_json(data);
								alert("JSON array", arr.stringify(4));
							}



					Operators:
						get_opIndex:

							Get a value out of the JSON array with literal syntax, for example `arr[2]`.

							var@ json_array::get_opIndex(uint index) property;

						Arguments:

							* uint index: the index of the value to get.

						Returns:

							var@: a handle to the object, or null if not found.

						Example:

							```NVGT
							void main() {
								json_array@ cats = parse_json("""["Athena", "Willow", "Punk", "Franky", "Yoda", "Waffles"]""");
								alert("The third cat is", cats[2]);
								for (uint i = 0; i < cats.size(); i++)
									alert("Awww", "Hi " + cats[i]);
							}



						opCall:

							Get a value out of a JSON array using a query.

							var@ json_array::opCall(string query);

						Arguments:

							* string query: the JSON query (see the remarks section for details).

						Returns:

							var@: a handle to the object, or null if it couldn't be found.

						Remarks:

							Queries are formatted like so, using key names and indexes separated with ".":

							> world.player.0

							It can be as nested as you like.

						Example:

							```NVGT
							void main() {
								string data = """[[["Colorado", "Kansas", "Minnesota"]]]""";
								json_array@ arr = parse_json(data);
								string location = arr("0.0.0");
								alert("My favorite state is", location);
							}



						set_opIndex:

							Set a value in the JSON array at a particular position.

							void json_array::set_opIndex(uint index, var@ value) property;

						Arguments:

							* uint index: the index to insert the item at.

							* var@ value: the value to set.

						Example:

							```NVGT
							void main() {
								json_array@ arr = parse_json("[]");
								arr[0] = "meow";
								alert("Info", "Cats say " + arr[0]);
							}



					Properties:
						empty:

							Determine if the JSON array is empty.

							const bool json_array::empty;

						Example:

							```NVGT
							void main() {
								json_array@ arr = parse_json("[]");
								json_array@ arr2 = parse_json("[1, 2, 3]");
								alert("First is empty", arr.empty ? "true" : "false");
								alert("Second is empty", arr2.empty ? "true" : "false");
							}



						escape_unicode:
							Determines the amount of escaping that occurs in JSON parsing.

							`bool json_array::escape_unicode;`

						Remarks:
							If this property is true, escaping will behave like normal. If it's false, only the characters absolutely vital to JSON parsing are escaped.


				json_object:
					Methods:
						clear:

							Clears the JSON object completely.

							void json_object::clear();

						Example:

							```NVGT
							void main() {
								string data = """{"numbers": [1, 2, 3, 4, 5]}""";
								json_object@ o = parse_json(data);
								alert("Before being cleared", o.stringify());
								o.clear();
								alert("After being cleared", o.stringify());
							}



						exists:

							Determine if a key exists in the JSON object.

							bool json_object::exists(const string?&in key);

						Arguments:

							* const string&in key: the key to search for.

						Returns:

							bool: true if the key exists, false if not.

						Example:

							```NVGT
							void main() {
								string data = """{"engine": "NVGT"}""";
								json_object@ o = parse_json(data);
								alert("Info", (o.exists("engine") ? "The key exists" : "The key does not exist"));
							}



						get_keys:

							Get a list of all the keys in the JSON object.

							string[]@ json_object::get_keys();

						Returns:

							string[]@: a handle to an array of strings containing all the keys of the JSON object.

						Remarks:

							Note that this function isn't recursive, you only get the keys for the object you're running this function on, not any child objects.

						Example:

							```NVGT
							void main() {
								string data = """{"thing": 1, "other_thing": "test", "another": true}""";
								json_object@ o = parse_json(data);
								string[]@ keys = o.get_keys();
								alert("The keys are", join(keys, ", "));
							}



						is_array:

							Determine if the value with the given key is a JSON array or not.

							bool json_object::is_array(string key);

						Arguments:

							* string key: the key to query.

						Returns:

							bool: true if the value with the specified key is a JSON array, false otherwise.

						Example:

							```NVGT
							void main() {
								string data = """{"classes": ["json_object", "json_array"]}""";
								json_object@ o = parse_json(data);
								alert("Info", o.is_array("classes") ? "array": "nonarray");
							}



						is_null:

							Determine if the value with the given key is null.

							bool json_object::is_null(string key);

						Arguments:

							* string key: the key to query.

						Returns:

							bool: true if the value with the specified key is null.

						Example:

							```NVGT
							void main() {
								string data = """{"brain": null}""";
								json_object@ o = parse_json(data);
								alert("Info", o.is_null("brain") ? "null" : "not null");
							}



						is_object:

							Determine if the value with the given key is a JSON object or not.

							bool json_object::is_object(string key);

						Arguments:

							* string key: the key to query.

						Returns:

							bool: true if the value with the specified key is a JSON object, false otherwise.

						Example:

							```NVGT
							void main() {
								string data = """{"json_object": {}}""";
								json_object@ o = parse_json(data);
								alert("Info", o.is_object("json_object") ? "Object" : "Non-object");
							}



						remove:

							Remove a value from the JSON object given a key.

							void json_object::remove(const string&in key);

						Arguments:

							* const string&in key: the key of the value to remove.

						Example:

							```NVGT
							void main() {
								string data = """{"name": "Quin"}""";
								json_object@ o = parse_json(data);
								alert("Initial", o.stringify());
								o.remove("name");
								alert("After removal", o.stringify());
							}



						set:

							Set a value in a JSON object.

							void json_object::set(const string&in key, var@ value);

						Arguments:

							* const string&in key: the key to give the value.

							* var@ value: a handle to the value to be set.

						Example:

							```NVGT
							void main() {
								json_object@ o = parse_json("{}");;
								o.set("nvgt_user", true);
								alert("Info", (bool(o["nvgt_user"]) ? "You are an NVGT user" : "You are not a regular NVGT user"));
							}



						size:

							Return the size of the JSON object.

							uint json_object::size();

						Remarks:

							Note that this function returns the number of top-level keys; it's not recursive.

						Example:

							```NVGT
							void main() {
								string data = """{"numbers": [1, 2, 3, 4, 5]}""";
								json_object@ o = parse_json(data);
								alert("Info", "The json object's size is " + o.size());
							}



						stringify:

							Return the JSON object as a string.

							string json_object::stringify(int indent = 0, int step = -1);

						Arguments:

							* int indent = 0: specifies how many spaces to use to indent the JSON. If 0, no indentation or formatting is performed.

							* int step = -1: the outwards indentation step, for example setting it to two would cause your outdents to only be two spaces. -1 (the default) keeps it how it is, normally you don't need to use this parameter.

						Returns:

							string: the JSON object as a string.

						Example:

							```NVGT
							void main() {
								string data = """{"name": "Quin", "age": 18}""";
								json_object@ o = parse_json(data);
								alert("JSON object", o.stringify(4));
							}



					Operators:
						get_opIndex:

							Get a value out of the JSON object with literal syntax, for example `json["key"]`.

							var@ json_object::get_opIndex(const string&in key) property;

						Arguments:

							* const string&in key: the key of the object to get.

						Returns:

							var@: a handle to the object, or null if not found.

						Example:

							```NVGT
							void main() {
								string data = """{"name": "Quin", "age": 18}""";
								json_object@ o = parse_json(data);
								alert("Info", o["name"] + " is " + o["age"]);
							}



						opCall:

							Get a value out of a JSON object using a query.

							var@ json_object::opCall(const string&in query);

						Arguments:

							* const string&in query: the JSON query (see the remarks section for details).

						Returns:

							var@: a handle to the object, null if not found.

						Remarks:

							Queries are formatted like so:

							> first_field.subfield.subfield

							It can be as nested as you like.

							This method also works on json_array's, and you access array indexes just with their raw numbers, like so:

							> world.players.0

						Example:

							```NVGT
							void main() {
								string data = """{"countries":{"us":["Colorado", "Kansas", "Minnesota"]}}""";
								json_object@ o = parse_json(data);
								string locations = o("countries.us");
								alert("My favorite states are", locations);
							}



						set_opIndex:

							Set a value in the JSON object with a particular key.

							void json_object::set_opIndex(const string&in key, var@ value) property;

						Arguments:

							* const string&in key: the key of the object.

							* var@ value: the value to set.

						Example:

							```NVGT
							void main() {
								json_object@ o = parse_json("{}");
								o["name"] = "Quin";
								alert("Hi", "I am " + o["name"]);
							}



					Properties:
						escape_unicode:
							Determines the amount of escaping that occurs in JSON parsing.

							`bool json_object::escape_unicode;`

						Remarks:
							If this property is true, escaping will behave like normal. If it's false, only the characters absolutely vital to JSON parsing are escaped.


				pack:
					methods:
						add_file:
							Add a file on disk to a pack.

							`bool pack::add_file(const string&in disk_filename, const string&in pack_filename, bool allow_replace = false);`

						Arguments:
							* const string&in disk_filename: the filename of the file to add to the pack.
							* const string&in pack_filename: the name the file should have in your pack.
							* bool allow_replace = false: if a file already exists in the pack with that name, should it be overwritten?

						Returns:
							bool: true if the file was successfully added, false otherwise.


						add_memory:
							Add content stored in memory to a pack.

							`bool pack::add_memory(const string&in pack_filename, const string&in data, bool replace = false);`

						Arguments:
							* const string&in pack_filename: the name the file should have in your pack.
							* const string&in data: a string containing the data to be added.
							* bool allow_replace = false: if a file already exists in the pack with that name, should it be overwritten?

						Returns:
							bool: true if the data was successfully added, false otherwise.


						close:
							Closes a pack, freeing all its resources.

							`bool pack::close();`

						Returns:
							bool: true if the pack was successfully closed, false otherwise.


						delete_file:
							Attempt to remove a file from a pack.

							`bool pack::delete_file(const string&in filename);`

						Arguments:
							* const string&in filename: the name of the file to be deleted.

						Returns:
							bool: true if the file was successfully deleted, false otherwise.


						file_exists:
							Query whether or not a file exists in your pack.

							`bool pack::file_exists(const string&in filename);`

						Arguments:
							* const string&in filename: the name of the file to query.

						Returns:
							bool: true if the file exists in the pack, false if not.


						get_file_name:
							Get the name of a file with a particular index.

							`string pack::get_file_name(int index);`

						Arguments:
							* int index: the index of the file to retrieve (see remarks).

						Returns:
							string: the name of the file if found, or an empty string if not.

						Remarks:
							The index you pass to this function is the same index you'd use to access an element in the return value from `pack::list_files()`.


						get_file_offset:
							Get the offset of a file in your pack.

							`uint pack::get_file_offset(const string&in filename);`

						Arguments:
							* const string&in filename: the name of the file to get the offset of.

						Returns:
							uint: the offset of the file (in bytes).

						Remarks:
							Do not confuse this with the offset parameter in the pack::read_file method. This function is provided encase you wish to re-open the pack file with your own file object and seek to a file's data within that external object. The offset_in_file parameter in the read_file method is relative to the file being read.


						get_file_size:
							Returns the size of a particula file in the pack.

							`uint pack::get_file_size(const string&in filename);`

						Arguments:
							* const string&in filename: the name of the file to query the size of.

						Returns:
							uint: the size of the file (in bytes), or 0 if no file with that name was found.


						list_files:
							Get a list of all the files in the pack.

							`string[]@ pack::list_files();`

						Returns:
							string[]@: a handle to an array of strings containing the names of every file in the pack.


						open:
							Open a pack to perform operations on it.

							`bool pack::open(const string&in filename, uint mode, bool memload = false);`

						Arguments:
							* const string&in filename: the name of the pack file to open.
							* uint mode: the mode to open the pack in (see `pack_open_modes` for more information).
							* bool memload = false: whether or not the pack should be loaded into memory as opposed to on disk.

						Returns:
							bool: true if the pack was successfully opened with the given mode, false otherwise.

						Remarks:
							To open an embedded pack, prefix it with `*`, like this:

							`pack.open("*embedded.dat");`

							For BGT compatibility, you can use a single `*` to open the first available embedded pack.

							`pack.open("*");`


						read_file:
							Get the contents of a file contained in a pack.

							`string pack::read_file(const string&in pack_filename, uint offset_in_file, uint size);`

						Arguments:
							* const string&in pack_filename: the name of the file to be read.
							* uint offset_in_file: the offset within the file to begin reading data from (do not confuse this with pack::get_file_offset)
							* uint size: the number of bytes to read (see `pack::get_file_size` to read the entire file).

						Returns:
							string: the contents of the file.

						Remarks:
							This function allows you to read chunks of data from a file, as well as an entire file in one call. To facilitate this, the offset and size parameters are provided. offset is relative to the file being read E. 0 for the beginning of it. So to read a file in one chunk, you could execute:
							string file_contents = my_pack.read_file("filename", 0, my_pack.get_file_size("filename"));


						set_pack_identifier:
							Set the identifier of this pack object (e.g. the first 8-bytes that determine if you have a valid pack or not).

							`bool pack::set_pack_identifier(const string&in ident);`

						Arguments:
							* const string&in ident: the new identifier (see remarks).

						Returns:
							bool: true if the pack's identifier was properly set, false otherwise.

						Remarks:
							* The default pack identifier is "NVPK" followed by 4 NULL bytes.
							* Your pack identifier should be 8 characters or less. If it's less than 8 characters, 0s will be added as padding on the end. If it's larger than 8, the first 8 characters will be used.


				regexp:

					The regexp object allows for the easy usage of regular expressions in NVGT.

					regexp(const string&in pattern, int options = RE_UTF8);

				Arguments:

					* const string&in pattern: the regular expression's pattern (see remarks).

					* int options = RE_UTF8: a combination of any of the values from the `regexp_options` enum.

				Remarks:

					Regular expressions are a language used for matching, substituting, and otherwise manipulating text. To learn about the regular expression syntax that nVGT uses (called PCRE), see this link: https://en.wikipedia.org/wiki/Perl_Compatible_Regular_Expressions

				Example:

					```NVGT
					void main() {
						regexp re("^[a-z]{2,4}$"); // Will match any lowercase alphabetical string between 2 and 4 characters in length.
						string compare = input_box("Text", "Enter the text to check against the regular expression.");
						if (compare.empty()) {
							alert("Regular expression tester", "You didn't type a string to test.");
							exit(1);
						}
						bool matches = re.match(compare);
						alert("The regular expression", matches ? "matches" : "does not match");
					}



					methods:
						match:
							Determine if the regular expression matches against a particular string or not.

							1. `bool regexp::match(const string&in subject, uint64 offset = 0);`
							2. `bool regexp::match(const string&in subject, uint64 offset, int options);`

						Arguments (1):
							* const string&in subject: the string to compare against.
							* uint64 offset = 0: the offset to start the comparison at.

						Arguments (2):
							* const string&in subject: the string to compare against.
							* uint64 offset: the offset to start the comparison at.
							* int options: any combination of the values found in the `regexp_options` enum.

						Returns:
							bool: true if the regular expression matches the given string, false if not.


			Enums:
				pack_open_modes:
					This is a list of all the possible open modes to be passed to the `pack::open()` function.

					* PACK_OPEN_MODE_NONE.
					* PACK_OPEN_MODE_APPEND: open the pack and append data to it.
					* PACK_OPEN_MODE_CREATE: create a new pack.
					* PACK_OPEN_MODE_READ: open the pack for reading.


				regexp_options:
					This enum holds various constants that can be passed to the regular expression classes in order to change their behavior.

				Notes:
					Portions of this regexp_options documentation were Copied from POCO header files.
					* Options marked [ctor] can be passed to the constructor of regexp objects.
					* Options marked [match] can be passed to match, extract, split and subst.
					* Options marked [subst] can be passed to subst.

					See the PCRE documentation for more information.

				Constants:
					* RE_CASELESS: case insensitive matching (/i) [ctor]
					* RE_MULTILINE: enable multi-line mode; affects ^ and $ (/m) [ctor]
					* RE_DOTALL: dot matches all characters, including newline (/s) [ctor]
					* RE_EXTENDED: totally ignore whitespace (/x) [ctor]
					* RE_ANCHORED: treat pattern as if it starts with a ^ [ctor, match]
					* RE_DOLLAR_END_ONLY: dollar matches end-of-string only, not last newline in string [ctor]
					* RE_EXTRA: enable optional PCRE functionality [ctor]
					* RE_NOT_BOL: circumflex does not match beginning of string [match]
					* RE_NOT_EOL: $ does not match end of string [match]
					* RE_UNGREEDY: make quantifiers ungreedy [ctor]
					* RE_NOT_EMPTY: empty string never matches [match]
					* RE_UTF8: assume pattern and subject is UTF-8 encoded [ctor]
					* RE_NO_AUTO_CAPTURE: disable numbered capturing parentheses [ctor, match]
					* RE_NO_UTF8_CHECK: do not check validity of UTF-8 code sequences [match]
					* RE_FIRSTLINE: an unanchored pattern is required to match before or at the first newline in the subject string, though the matched text may continue over the newline [ctor]
					* RE_DUPNAMES: names used to identify capturing subpatterns need not be unique [ctor]
					* RE_NEWLINE_CR: assume newline is CR ('\r'), the default [ctor]
					* RE_NEWLINE_LF: assume newline is LF ('\n') [ctor]
					* RE_NEWLINE_CRLF: assume newline is CRLF ("\r\n") [ctor]
					* RE_NEWLINE_ANY: assume newline is any valid Unicode newline character [ctor]
					* RE_NEWLINE_ANY_CRLF: assume newline is any of CR, LF, CRLF [ctor]
					* RE_GLOBAL: replace all occurrences (/g) [subst]
					* RE_NO_VARS: treat dollar in replacement string as ordinary character [subst]


			Functions:
				ascii_to_character:

					Return the character that corresponds to the given ascii value.

					string ascii_to_character(uint8 ascii);

				Arguments:

					* uint8 ascii: the ascii value to convert.

				Returns:

					string: the character for the given ascii value.

				Example:

					```NVGT
					void main() {
						uint8 ascii = parse_int(input_box("ASCII value", "Enter the ascii value to convert."));
						alert("Info", ascii + " has a value of " + ascii_to_character(ascii));
					}



				character_to_ascii:

					Return the ascii value for the given character.

					uint8 character_to_ascii(const string&in character);

				Arguments:

					* const string&in character: the character to convert.

				Returns:

					uint8: the ascii value for the given character.

				Remarks:

					If a string containing more than one character is passed as input to this function, only the left-most character is checked.

					If an empty string is passed, the function returns 0.

				Example:

					```NVGT
					void main() {
						string character = input_box("Character", "Enter a character to convert");
						if (character.is_empty()) {
							alert("Error", "You did not type anything.");
							exit();
						}
						if (character.length() != 1) { // The user typed more than one character.
							alert("Error", "You must only type a single character.");
							exit();
						}
						alert("Info", character + " has an ascii value of " + character_to_ascii(character));
					}



				join:

					turn an array into a string, with each element being separated by the given delimiter.

					string join(const string[]@ elements, const string&in delimiter);

				Arguments:

					* const string[]@ elements: a handle to the array to join.

					* const string&in delimiter: the delimiter used to separate each element in the array (can be empty).

				Returns:

					string: the given array as a string.

				Remarks:

					If an empty array is passed, a blank string is returned.

					The delimiter is not appended to the end of the last item in the returned string. For example if an array contains [1, 2, 3] and you join it with a delimiter of . (period), the result would be "1.2.3" without an extra delimiter appended.

				Example:

					```NVGT
					void main() {
						string[] names = {"Sam", "Quin", "Patrick"};
						string people = join(names, ", ");
						alert("Info", people);
					}



				number_to_words:

					Convert a number into its string equivalent, for example, one thousand four hundred and fifty six.

					string number_to_words(int64 the_number, bool include_and = true);

				Arguments:

					* int64 the_number: the number to convert.

					* bool include_and = true: whether or not to include the word "and" in the output.

				Returns:

					string: the specified number, converted to a readable string.

				Remarks:

					At this time, this function only produces English results.

				Example:

					```NVGT
					void main() {
						int64 num = random(1000, 100000);
						string result = number_to_words(num);
						alert("Info", num + " as a string is " + result);
					}



				pack_set_global_identifier:
					Set the global identifier of all your packs (e.g. the first 8-bytes that determine if you have a valid pack or not).

					`bool pack_set_global_identifier(const string&in ident);`

				Arguments:
					* const string&in ident: the new identifier (see remarks).

				Returns:
					bool: true if the identifier was properly set, false otherwise.

				Remarks:
					* The default pack identifier is "NVPK" followed by 4 NULL bytes.
					* Your pack identifier should be 8 characters or less. If it's less than 8 characters, 0s will be added as padding on the end. If it's larger than 8, the first 8 characters will be used.
					* NVGT will refuse to open any pack files that do not match this identifier.


				regexp_match:

					Test if the text matches the specified regular expression.

					bool regexp_match(const string&in text, const string&in pattern);

				Arguments:

					* const string&in text: the text to compare against the regular expression.

					* const string&in pattern: the regular expression to match.

				Returns:

					bool: true if the text matches the given regular expression, false otherwise.

				Remarks:

					If you would like a lower level interface to regular_expressions in NVGT, check out the regexp class which this function more or less wraps.

					To learn more about regular expressions, visit: https://en.wikipedia.org/wiki/Regular_expression

				Example:

					```NVGT
					void main() {
						bool match = regexp_match("Test", "^[a-zA-Z]+$");
						alert("the regular expression", match ? "matches" : "does not match");
					}



				string_base32_decode:

					Decodes a string from base32.

					string string_base32_decode(const string&in the_data);

				Arguments:

					* const string&in the_data: The data that is to be decoded.

				Returns:

					String: The decoded string on success or an empty string on failure.

				Remarks:

					You can learn more about the base32 format [here](https://en.wikipedia.org/wiki/Base32).

				Example:

					```NVGT
					void main() {
						string text = input_box("Text", "Enter the text to decode.");
						if (text.is_empty()) {
							alert("Error", "You did not type any text.");
							exit();
						}
						alert("Info", string_base32_decode(text));
					}



				string_base32_encode:

					Encodes a string as base32.

					string string_base32_encode(const string&in the_data);

				Arguments:

					* const string&in the_data: The data that is to be encoded.

				Returns:

					String: The encoded string on success or an empty string on failure.

				Remarks:

					You can learn more about the base32 format [here](https://en.wikipedia.org/wiki/Base32).

				Example:

					```NVGT
					void main() {
						string text = input_box("Text", "Enter the text to encode.");
						if (text.is_empty()) {
							alert("Error", "You did not type any text.");
							exit();
						}
						alert("Info", string_base32_encode(text));
					}



				string_base32_normalize:

					Normalize a string to conform to the base32 spec.

					string string_base32_normalize(const string&in the_data);

				Arguments:

					* const string&in the_data: the string to normalize.

				Returns:

					string: the normalized string.

				Remarks:

					The primary thing this function does is to skip separators, convert all letters to uppercase, and make sure the length is a multiple of 8.

					To learn more about base32, click [here](https://en.wikipedia.org/wiki/Base32).

				Example:

					```NVGT
					void main() {
						string text = input_box("Text", "Enter the text to normalize.");
						if (text.is_empty()) {
							alert("Error", "You didn't type anything.");
							exit();
						}
						alert("Info", string_base32_normalize(text));
					}



				string_base64_decode:

					Decodes a string which has previously been encoded in base64.

					string string_base64_decode(const string&in the_data, string_base64_options options = STRING_BASE64_PADLESS);

				Arguments:

					* const string&in the_data: The data that is to be decoded.

					* string_base64_options options = STRING_BASE64_PADLESS: A bitwise of options that control the behavior of the decoding (see remarks).

				Returns:

					String: The decoded string on success or an empty string on failure.

				Remarks:

					The following options may be passed to the options argument of this function:

					* STRING_BASE64_DEFAULT: The defaults passed to the string_base64_encode function.

					* STRING_BASE64_URL: Use base64 URL encoding as opposed to the / and = characters.

					* STRING_BASE64_PADLESS: Allows decoding even if the string does not end with = padding characters.

					You can learn more about base64 [here.](https://en.wikipedia.org/wiki/Base64)

				Example:

					```NVGT
					void main() {
						alert("example", string_base64_decode("aGVsbG8=")); // Should print "hello".
					}



				string_base64_encode:

					Encodes a string as base64.

					string string_base64_encode(const string&in the_data, string_base64_options options = STRING_BASE64_DEFAULT);

				Arguments:

					* const string&in the_data: The data that is to be encoded.

					* string_base64_options options = STRING_BASE64_DEFAULT: A bitwise of options that control the behavior of the encoding (see remarks).

				Returns:

					String: The encoded string on success or an empty string on failure.

				Remarks:

					The following options may be passed to the options argument of this function:

					* STRING_BASE64_DEFAULT: The default options (i.e. insert padding and do not use URL encoding).

					* STRING_BASE64_URL: Use base64 URL encoding as opposed to the / and = characters.

					* STRING_BASE64_PADLESS: Allows decoding even if the string does not end with = padding characters.

					You can learn more about base64 [here.](https://en.wikipedia.org/wiki/Base64)

				Example:

					```NVGT
					void main() {
						alert("example", string_base64_encode("Hello")); // Should print SGVsbG8=
					}



			Global Properties:
				pack_global_identifier:
					Represents the identifier currently being used for all your packs.

					`const string pack_global_identifier;`


		Date and Time:
			Classes:
				datetime:
					This class stores an instance in time represented as years, months, days, hours, minutes, seconds, milliseconds and microseconds.

					The class stores time independent of timezone, and thus uses UTC by default. You can use a calendar object in place of this one if you need to check local time, however it is faster to calculate time without needing to consider the timezone and thus any time difference calculations should be done with this object instead of calendar.

					1. `datetime();`
					2. datetime(double julian_day);
					3. datetime(int year, int month, int day, int hour = 0, int minute = 0, int second = 0, int millisecond = 0, int microsecond = 0);

				Arguments (2):
					* double julian_day: The julian day to set the time based on.

				Arguments (3):
					* int year: the year to set.
					* int month: the month to set.
					* int day: the day to set.
					* int hour = 0: the hour to set (from 0 to 23).
					* int minute = 0: the minute to set (from 0 to 59).
					* int second = 0: the second to set (0 to 59).
					* int millisecond = 0: the millisecond to set (0 to 999).
					* int microsecond = 0: the microsecond to set (0 to 999).



					Methods:
						format:

							Formats the contents of a datetime object as a string given any number of format specifiers.

							string format(string fmt, int timezone_difference = UTC);

						Arguments:

							* string fmt: A string describing how the datetime object should be formatted (see remarks).

						Returns:

							string: The formatted date/time.

						Remarks:

							The date time formatter in NVGT comes from the Poco C++ libraries, the below remarks other than anything directly mentioning NVGT were copied verbatim from the Poco documentation.

							The format string is used as a template to format the date and is copied character by character except for the following special characters, which are replaced by the corresponding value.

							* %w - abbreviated weekday (Mon, Tue, ...)

							* %W - full weekday (Monday, Tuesday, ...)

							* %b - abbreviated month (Jan, Feb, ...)

							* %B - full month (January, February, ...)

							* %d - zero-padded day of month (01 .. 31)

							* %e - day of month (1 .. 31)

							* %f - space-padded day of month ( 1 .. 31)

							* %m - zero-padded month (01 .. 12)

							* %n - month (1 .. 12)

							* %o - space-padded month ( 1 .. 12)

							* %y - year without century (70)

							* %Y - year with century (1970)

							* %H - hour (00 .. 23)

							* %h - hour (00 .. 12)

							* %a - am/pm

							* %A - AM/PM

							* %M - minute (00 .. 59)

							* %S - second (00 .. 59)

							* %s - seconds and microseconds (equivalent to %S.%F)

							* %i - millisecond (000 .. 999)

							* %c - centisecond (0 .. 9)

							* %F - fractional seconds/microseconds (000000 - 999999)

							* %z - time zone differential in ISO 8601 format (Z or +NN.NN)

							* %Z - time zone differential in RFC format (GMT or +NNNN)

							* %% - percent sign

						Example:

							```NVGT
							void main() {
								datetime dt;
								alert("example", "The current date/time is " + dt.format("%Y/%m/%d, %h:%M:%S %A %Z"));
							}



						reset:

							Resets this datetime object to the current date and time.

							void reset();

						Remarks:

							The expression d.reset() is equivalent to the expression d = datetime(); this is only a convenience function.

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(2024, 3, 9, 1, 24, 49);
								alert("The datetime object is currently set to", d.year + "/" + d.month + "/" + d.day + ", " + d.hour + ":" + d.minute + ":" + d.second);
								d.reset();
								alert("The datetime object is now set to", d.year + "/" + d.month + "/" + d.day + ", " + d.hour + ":" + d.minute + ":" + d.second);
							}



						set:

							Set the date and time of the datetime object.

							datetime& datetime::set(int year, int month, int day, int hour = 0, int minute = 0, int second = 0, int millisecond = 0, int microsecond = 0);

						Arguments:

							* int year: the year to set.

							* int month: the month to set.

							* int day: the day to set.

							* int hour = 0: the hour to set (from 0 to 23).

							* int minute = 0: the minute to set (from 0 to 59).

							* int second = 0: the second to set (0 to 59).

							* int millisecond = 0: the millisecond to set (0 to 999).

							* int microsecond = 0: the microsecond to set (0 to 999).

						Returns:

							datetime&: A reference to the datetime object allowing for chaining calls together.

						Remarks:

							This method will throw an exception if the date or time is invalid. You can use the datetime_is_valid global function which takes similar arguments to this one and returns a boolean if you want to verify any values before passing them to this method.

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(2024, 3, 9, 1, 24, 49);
								alert("The datetime object is currently set to", d.year + "/" + d.month + "/" + d.day + ", " + d.hour + ":" + d.minute + ":" + d.second);
							}



						week:

							Returns the currently set week number within the year of the datetime object.

							int week(int first_day_of_week = 1);

						Arguments:

							* int first_day_of_week = 1: Determines whether 0 (sunday) or 1 (monday) begins the week.

						Returns:

							int: The week number within the year (0 to 53).

						Remarks:

							The first_day_of_week argument controls whether the week counter increases by 1 upon reaching Sunday or Monday. Passing other values to this function has undefined results.

							Week 1 of the year begins in whatever week contains January 4th.

						Example:

							```NVGT
							void main() {
								datetime dt(2016, 9, 17);
								alert("example", "week " + dt.week()); // Will display 37.
							}



					Operators:
						opAdd:
							Return a new datetime object with a duration of time represented as a timespan added to it.
							`datetime opAdd(const timespan&in);`


						opAddAssign:
							Add a duration represented as a timespan to a datetime object.
							`datetime& opAddAssign(const timespan&in);`


						opCmp:
							Compare a datetime object relationally to another one.
							`int opCmp(const datetime& other);`


						opEquals:
							Check if a datetime object is equal to another datetime object.
							`bool opEquals(const datetime& other);`


						opSub:
							1. Return a new datetime object with a duration of time represented as a timespan subtracted from it.
							2. Return a duration represented by a timespan by subtracting 2 datetime objects from each other.
							1. `datetime opSub(const timespan&in);`
							2. `timespan opSub(const datetime&in);`


						opSubAssign:
							Subtract a duration represented as a timespan from a datetime object.
							`datetime& opSubAssign(const timespan&in);`


					Properties:
						AM:

							A boolean which is true if the datetime object is set to an hour before noon (ante meridiem).

							const bool AM;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(2024, 5, 8, 14, 18);
								alert("The datetime's AM value was set to", d.AM);
							}



						day:

							Represents the current day of the datetime object.

							const uint day;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(2024, 5, 8);
								alert("The datetime's set day is", d.day);
							}



						hour:

							Represents the current hour (from 0 to 23) of the datetime object.

							const uint hour;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(d.year, d.month, d.day, 13, 47, 19);
								alert("The set hour of the datetime object is", d.hour);
							}



						hour12:

							Represents the current hour (from 0 to 12) of the datetime object.

							const uint hour12;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(d.year, d.month, d.day, 13, 47, 19);
								alert("In 12 hour time, the set hour of the datetime object is", d.hour12);
							}



						julian_day:

							Represents the current timestamp held by the datetime object in julian days.

							const double julian_day;

						Remarks:

							The [julian day](http://en.wikipedia.org/wiki/Julian_day) measures, including decimal fractions, the amount of time that has passed (in years) since Monday, January 1st, 4713 BC at 12 PM Universal Time.

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(2024, 5, 8, 16, 17, 18);
								alert("The julian day represented by this datetime object is", d.julian_day);
							}



						microsecond:

							Represents the current microsecond of the datetime object.

							const uint microsecond;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(d.year, d.month, d.day, 12, 47, 19, 217, 962);
								alert("The set microsecond value of the datetime object is", d.microsecond);
							}



						millisecond:

							Represents the current millisecond of the datetime object.

							const uint millisecond;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(d.year, d.month, d.day, 12, 47, 19, 217);
								alert("The set millisecond value of the datetime object is", d.millisecond);
							}



						minute:

							Represents the current minute of the datetime object.

							const uint minute;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(d.year, d.month, d.day, 12, 47, 19);
								alert("The set minute of the datetime object is", d.minute);
							}



						month:

							Represents the current month of the datetime object.

							const uint month;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(2024, 5, 8);
								alert("The datetime's set month is", d.month);
							}



						PM:

							A boolean which is true if the datetime object is set to an hour after noon (post meridiem).

							const bool PM;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(2024, 5, 8, 14, 18);
								alert("The datetime's set PM value is", d.PM);
							}



						second:

							Represents the current second of the datetime object.

							const uint second;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(d.year, d.month, d.day, 12, 47, 19);
								alert("The set second of the datetime object is", d.second);
							}



						timestamp:

							Represents the datetime object as a timestamp object.

							const timestamp timestamp;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(d.year, d.month, d.day, 12, 47, 19);
								alert("The current timestamp represented by the datetime object is", d.timestamp / SECONDS);
							}



						UTC_time:

							Determines the UTC based time point stored in the datetime object.

							const int64 UTC_time;

						Remarks:

							UTC based time begins on October 15, 1582 at 12 AM and uses an accuracy of 100 nanoseconds.

						Example:

							```NVGT
							void main() {
								datetime dt;
								alert("example", "The current UTC time value is " + dt.UTC_time);
							}



						weekday:

							Represents the current day of the week (from 0 to 6) calculated by the datetime object.

							const uint weekday;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(2024, 5, 8);
								alert("The datetime's set day of the week is", d.weekday);
							}



						year:

							Represents the current year of the datetime object.

							const uint year;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(2024, 5, 8);
								alert("The datetime's set year is", d.year);
							}



						yearday:

							Represents the current day of the year calculated by the datetime object.

							const uint yearday;

						Example:

							```NVGT
							void main() {
								datetime d;
								d.set(2024, 5, 8);
								alert("The datetime's set day of the year is", d.yearday);
							}



				timestamp:
					Stores a unix timestamp with microsecond accuracy and provides methods for comparing them.
					1. timestamp();
					2. timestamp(int64 epoch_microseconds);

				Arguments 2:
					* int64 epoch_microseconds: the initial value of the timestamp

				Remarks:
					If a timestamp object is initialize with the default constructor, it will be set to the system's current date and time.
					The unix epoch began on January 1st, 1970 at 12 AM UTC.


					Methods:
						has_elapsed:

							determine whether the given number of microseconds has elapsed since the time point stored in this timestamp.

							bool has_elapsed(int64 microseconds);

						Arguments:

							* int64 microseconds: the number of microseconds to check elapsed status of

						Returns:

							bool: Whether the given number of milliseconds has elapsed

						Remarks:

							This method serves as a shorthand version of executing `bool has_elapsed = timestamp() - this >= microseconds;`

						Example:

							```NVGT
							void main() {
								timestamp ts;
								alert("test", "will you keep the alert box opened for 10 seconds?");
								if (ts.has_elapsed(10 * SECONDS)) alert("good job", "You kept the alert box opened for 10 seconds!");
								else alert("oh no", "looks like a bit of character development in regards to the trait of patience is in order...");
							}



						update:

							Reset this timestamp to the system's current date and time.

							void update();

						Example:

							```NVGT
							void main() {
								timestamp ts;
								ts -= 60 * SECONDS;
								alert("timestamp before reset", ts / SECONDS);
								ts.update();
								alert("current timestamp", ts / SECONDS);
							}



					Operators:
						opImplConv:
							Implicitly convert the timestamp to a 64 bit integer containing a raw microsecond accuracy unix epoch.
							int64 opImplConv();


					Properties:
						elapsed:

							Determine the difference of time in microseconds between this timestamp's value and now.

							const int64 elapsed;

						Remarks:

							This property is a shorthand version of the expression `int64 elapsed = timestamp() - this;`

						Example:

							```NVGT
							void main() {
								datetime dt(2013, 11, 29, 1, 2, 3);
								alert("example", dt.format(DATE_TIME_FORMAT_RFC850) + " was " + (dt.timestamp.elapsed / DAYS) + " days ago");
							}



						UTC_time:

							Determines a UTC based time point based on the value stored in the timestamp object.

							const int64 UTC_time;

						Remarks:

							UTC based time begins on October 15, 1582 at 12 AM and uses an accuracy of 100 nanoseconds.

						Example:

							```NVGT
							void main() {
								timestamp ts;
								alert("example", "The current UTC time value is " + ts.UTC_time);
							}



			Functions:
				datetime_days_of_month:

					Returns the number of days in a given month, a year is also required so that leap year can be handled correctly.

					int datetime_days_of_month(int year, int month);

				Arguments:

					* int year: The year involved in this calculation due to leap year.

					* month: The month to fetch the number of days of (from 1 to 12).

				Returns:

					int: The number of days in the month provided with leap year accounted for, or 0 if invalid/out of range arguments are provided.

				Example:

					```NVGT
					void main() {
						for(int i = 1; i <= 12; i++) {
							alert("month " + i + " no leap year", datetime_days_of_month(2023, i));
							if (i == 2) alert("month " + i + " leap year", datetime_days_of_month(2024, i));
						}
						alert("invalid", datetime_days_of_month(2015, 17)); // Will show 0 (there are not 17 months in the year)!
					}



				datetime_is_leap_year:

					Determine whether a given year is a leap year.

					bool datetime_is_leap_year(int year);

				Arguments:

					* int year: The year to check the leap year status of.

				Returns:

					* bool: true if the given year is a leap year, false otherwise.

				Example:

					```NVGT
					void main() {
						for(int year = 2019; year < 2026; year ++) {
							alert("example", year+" is " + (datetime_is_leap_year(year)? "a leap year!" : "not a leap year."));
						}
					}



				datetime_is_valid:

					Check whether the components of a date/time are valid.

					bool datetime_is_valid(int year, int month, int day, int hour = 0, int minute = 0, int second = 0, int millisecond = 0, int microsecond = 0);

				Arguments:

					* int year: the year to check.

					* int month: the month to check (from 1 to 12).

					* int day: the day to check (from 1 to 31).

					* int hour = 0: the hour to check (from 0 to 23).

					* int minute = 0: the minute to check (from 0 to 59).

					* int second = 0: the second to check (0 to 59).

					* int millisecond = 0: the millisecond to check (0 to 999).

					* int microsecond = 0: the microsecond to check (0 to 999).

				Returns:

					bool: true if the given arguments make up a valid date and time, false otherwise.

				Example:

					```NVGT
					void main() {
						alert("example 1", datetime_is_valid(2023, 2, 29)); // Will display false (not a leap year!)
						alert("example 2", datetime_is_valid(2020, 2, 29)); // Will display true, is a leap year.
						alert("example 3", datetime_is_valid(2020, 6, 31)); // Will display false (June has 30 days).
						alert("example 4", datetime_is_valid(2020, 7, 31, 13, 71, 59)); // Will display false (invalid time values).
					}



				parse_datetime:

					Parse a string into a datetime object given a format specifier.

					1. datetime parse_datetime(string fmt, string datetime_to_parse, int& timezone_difference);

					2. datetime parse_datetime(string datetime_to_parse, int& timezone_difference);

				Arguments:

					* string fmt (1): The format specifier string.

					* string datetime_to_parse: The string containing a date/time to parse according to the given format specifier, or automatically if one is not provided.

					* int& timezone_difference: A reference to an integer which will store any timezone offset parsed from the provided string.

				Returns:

					datetime: A datetime object containing the parsed information.

				Remarks:

					This function may throw an exception if an invalid date/time is provided. You can use the global `bool datetime_is_valid_format(string str)` function to attempt validating user input before passing it as an argument here.

					If you use the version of this function without a format specifier, it will try to automatically determine the format of the date/time using several available standard formats.

					This function will clean the input string such as trimming whitespace before parsing is attempted.

					Internally, this wraps the DateTimeParser functionality in the Poco c++ libraries. Their documentation says the following, copied verbatim, about this parsing engine, though note in NVGT DateTime:::makeUTC is called datetime.make_UTC, for example.

					This class provides a method for parsing dates and times from strings. All parsing methods do their best to parse a meaningful result, even from malformed input strings.

					The returned DateTime will always contain a time in the same timezone as the time in the string. Call DateTime::makeUTC() with the timeZoneDifferential returned by parse() to convert the DateTime to UTC.

					Note: When parsing a time in 12-hour (AM/PM) format, the hour (%h) must be parsed before the AM/PM designator (%a, %A), otherwise the AM/PM designator will be ignored.

					See the DateTimeFormatter class for a list of supported format specifiers.

					In addition to the format specifiers supported by DateTimeFormatter, an additional specifier is supported: %r will parse a year given by either two or four digits. Years 69-00 are interpreted in the 20th century (1969-2000), years 01-68 in the 21th century (2001-2068).

					Note that in the current implementation all characters other than format specifiers in the format string are ignored/not matched against the date/time string. This may lead to non-error results even with nonsense input strings. This may change in a future version to a more strict behavior.

					If more strict format validation of date/time strings is required, a regular expression could be used for initial validation, before passing the string to DateTimeParser.

				Example:

					```NVGT
					void main() {
						int tzd;
						datetime dt = parse_datetime("2024-03-19 13:19:51", tzd);
						alert("example", "the parsed time is " + dt.format("%W, %B %d %Y at %H:%M:%S"));
						dt = parse_datetime("%Y/%m/%d %H:%M:%S", "2024/03/19 13:19:51", tzd);
						alert("example", "the parsed time is " + dt.format("%W, %B %d %Y at %H:%M:%S"));
					}



				timestamp_from_UTC_time:

					Create a timestamp object from a UTC time point.

					timestamp timestamp_from_UTC_time(int64 UTC_time);

				Arguments:

					* int64 UTC_time: The UTC time point (see remarks).

				Returns:

					timestamp: A timestamp derived from the given value.

				Remarks:

					UTC based time begins on October 15, 1582 at 12 AM and uses an accuracy of 100 nanoseconds.

				Example:

					```NVGT
					void main() {
						calendar c(2017, 2, 3, 4, 5, 6);
						alert("initial calendar", c.format(DATE_TIME_FORMAT_RFC1123));
						timestamp ts = timestamp_from_UTC_time(c.UTC_time);
						calendar c2 = ts;
						alert("new calendar", c2.format(DATE_TIME_FORMAT_RFC1123));
					}



			Global Properties:
				DATE_DAY:

					Returns the number of the current day through the month.

					const int DATE_DAY;

				Example:

					```NVGT
					void main() {
						alert("example", "It is day " + DATE_DAY + " of " + DATE_MONTH_NAME);
					}



				DATE_MONTH:

					Returns the month set on the system, from 1 to 12.

					const int DATE_MONTH;

				Example:

					```NVGT
					void main() {
						alert("example", "the month is " + DATE_MONTH);
					}



				DATE_MONTH_NAME:

					Returns the name of the current month.

					const string DATE_MONTH_NAME;

				Example:

					```NVGT
					void main() {
						alert("example", "the month is " + DATE_MONTH_NAME);
					}



				DATE_WEEKDAY:

					Returns the number of the current day through the week.

					const int DATE_WEEKDAY;

				Example:

					```NVGT
					void main() {
						alert("example", "It is day " + DATE_WEEKDAY + " of the week");
					}



				DATE_WEEKDAY_NAME:

					Returns the name of the current day.

					const string DATE_WEEKDAY_NAME;

				Example:

					```NVGT
					void main() {
						alert("example", "It is " + DATE_WEEKDAY_NAME + " today");
					}



				DATE_YEAR:

					Returns the 4 digit year set on the system.

					const int DATE_YEAR;

				Example:

					```NVGT
					void main() {
						alert("example", "the year is " + DATE_YEAR);
					}



				SCRIPT_BUILD_TIME:

					Contains the timestamp when the calling nvgt program was compiled.

					const timestamp SCRIPT_BUILD_TIME;

				Remarks:

					This property will be set to the time when NVGT launched if it is accessed by a script running from source. Otherwise, it will contain a timestamp set to when the .nvgt script was compiled into an executable.

				Example:

					```NVGT
					void main() {
						if (!SCRIPT_COMPILED) {
							alert("oops", "This only works in compiled scripts.");
							return;
						}
						alert("Script compiled on", datetime(SCRIPT_BUILD_TIME).format(DATE_TIME_FORMAT_RFC850));
					}



				TIME_HOUR:

					Returns the current hour, in 24-hour format.

					const int TIME_HOUR;

				Example:

					```NVGT
					void main() {
						alert("Example", "It is currently " + TIME_HOUR + ":00");
					}



				TIME_MINUTE:

					Returns the current minute.

					const int TIME_MINUTE;

				Example:

					```NVGT
					void main() {
						alert("Example", "It is currently " + TIME_HOUR + ":" + TIME_MINUTE);
					}



				TIME_SECOND:

					Returns the current second.

					const int TIME_SECOND;

				Example:

					```NVGT
					void main() {
						alert("Example", "It is currently " + TIME_HOUR + ":" + TIME_MINUTE + ":" + TIME_SECOND);
					}



				TIME_SYSTEM_RUNNING_MILLISECONDS:

					Represents the number of milliseconds since the system booted up.

					uint64 TIME_SYSTEM_RUNNING_MILLISECONDS;

				Remarks:

					Note that unlike something like `GetTickCount()` from the Windows API, this value is given as an unsigned 64-bit integer, so it would take an almost impossible amount of uptime to make it overflow.

				Example:

					```NVGT
					void main() {
						alert("Your computer has been up for", TIME_SYSTEM_RUNNING_MILLISECONDS + " MS");
					}



				timer_default_accuracy:

					Set or retrieve the default accuracy used for timers.

					uint64 timer_default_accuracy = MILLISECONDS;

				Remarks:

					Internally NVGT always uses a clock with microsecond accuracy to track time, meaning that microseconds is the highest accuracy that timer objects support. However in many cases, microsecond accuracy is not desired and causes a user to need to type far too many 0s than they would desire when handling elapsed time. Therefor NVGT allows you to set a devisor/multiplier that is applied to any timer operation as it passes between NVGT and your script.

					The accuracy represented here is just a number of microseconds to divide values returned from timer.elapsed by before returning them to your script, as well as the number of microseconds to multiply by if one uses the timer.force function.

					If this property is changed, any timers that already exist will remain unaffected, and only newly created timers will use the updated value.

					Some useful constants are provided to make this property more useful:

					* MILLISECONDS: 1000 microseconds

					* SECONDS: 1000 MILLISECONDS

					* MINUTES: 60 SECONDS

					* HOURS: 60 MINUTES

					* DAYS: 24 HOURS.

				Example:

					```NVGT
					void main() {
						timer_default_accuracy = SECONDS;
						alert("example", "about to wait for 1100ms");
						timer t;
						wait(1100);
						alert("example", t.elapsed); // Will show 1.
						timer_default_accuracy = 250; // You can set this to arbitrary values if you so wish.
						timer t2;
						wait(1); // or 1000 microseconds.
						alert("example", t2.elapsed); // Will display a number slightly higher than 4 as the wait call takes some extra microseconds to return.
					}



				TIMEZONE_BASE_OFFSET:

					Determine the system's timezone offset from UTC in seconds, excluding any daylight saving time alterations.

					const int TIMEZONE_BASE_OFFSET;

				Example:

					```NVGT
					void main() {
						alert(TIMEZONE_NAME, "base offset is " + TIMEZONE_BASE_OFFSET);
					}



				TIMEZONE_DST_NAME:

					Retrieve the name of the timezone currently set on the system that is displayed when daylight saving time is in effect.

					const string TIMEZONE_STANDARD_NAME;

				Remarks:

					This will always return the version of the timezone name that is shown when daylight saving time is active, even if daylight saving time is not actually in effect.

					If the current timezone does not use daylight saving time, the value of this property is somewhat undefined but is most likely to equal TIMEZONE_STANDARD_NAME. The undefined value condition is a result of directly querying the operating system's API for this information, different platforms might react in slightly different ways to the absants of daylight savings support in a timezone.

				Example:

					```NVGT
					void main() {
						alert("timezone example", "Currently in " + TIMEZONE_NAME + ", which is usually called " + TIMEZONE_STANDARD_NAME + " though it is known as " + TIMEZONE_DST_NAME + "when daylight saving time is in effect.");
					}



				TIMEZONE_DST_OFFSET:

					Determine the DST alteration offset in seconds if daylight saving time is active in the system's current timezone.

					const int TIMEZONE_DST_OFFSET;

				Remarks:

					If daylight saving time usually moves the clock an hour forward in your timezone when it is active, for example, this value will return 3600.

				Example:

					```NVGT
					void main() {
						timespan t(TIMEZONE_DST_OFFSET, 0);
						alert(TIMEZONE_NAME, "The clock is being altered by " + t.hours + " hours, " + t.minutes + " minutes, and " + t.seconds + " seconds due to daylight saving time.");
					}



				TIMEZONE_NAME:

					Retrieve the name of the timezone currently set on the system.

					const string TIMEZONE_NAME;

				Remarks:

					This contains the value of either TIMEZONE_STANDARD_NAME or TIMEZONE_DST_NAME, depending on whether daylight saving time is currently in effect. If the timezone does not support daylight saving time, this always yields TIMEZONE_STANDARD_NAME.

				Example:

					```NVGT
					void main() {
						alert("timezone example", "Currently in " + TIMEZONE_NAME + ", which is usually called " + TIMEZONE_STANDARD_NAME + " though it is known as " + TIMEZONE_DST_NAME + "when daylight saving time is in effect.");
					}



				TIMEZONE_OFFSET:

					Determine the system's current timezone offset from UTC in seconds.

					const int TIMEZONE_OFFSET;

				Remarks:

					This is also the sum of TIMEZONE_BASE_OFFSET + TIMEZONE_DST_OFFSET, provided for convenience.

				Example:

					```NVGT
					void main() {
						alert(TIMEZONE_NAME, "offset from UTC: " + timespan(TIMEZONE_OFFSET, 0).format());
					}



				TIMEZONE_STANDARD_NAME:

					Retrieve the name of the timezone currently set on the system that is displayed when daylight saving time is not in effect.

					const string TIMEZONE_STANDARD_NAME;

				Remarks:

					This will always return the version of the timezone name that is shown when daylight saving time is not active, even if daylight saving time is truly in effect.

				Example:

					```NVGT
					void main() {
						alert("timezone example", "Currently in " + TIMEZONE_NAME + ", which is usually called " + TIMEZONE_STANDARD_NAME + " though it is known as " + TIMEZONE_DST_NAME + "when daylight saving time is in effect.");
					}



		Environment:
			In this reference, the environment section specifically refers to the script/system environment that your program is being run on. Examples could include the current line number in the nvgt script being compiled, the operating system being run on or fetching the value of a shell environment variable.


			Enums:
				system_power_state:
					This is a list of possible values returned from the `system_power_info()` function.

					* POWER_STATE_ERROR: an error was encountered.
					* POWER_STATE_UNKNOWN: the power state is unknown.
					* POWER_STATE_ON_BATTERY: the system is running on  battery power.
					* POWER_STATE_NO_BATTERY: the system has no battery installed.
					* POWER_STATE_CHARGING: the battery is currently recharging.
					* POWER_STATE_CHARGED: the battery is fully charged, but still actively plugged in.


			Functions:
				chdir:

					Change the current working directory.

					bool chdir(const string&in new_directory);

				Arguments:

					* const string&in new_directory: The directory you'd like to change into. Note that this directory must already exist.

				Returns:

					bool: true if NVGT could set the current working directory to the specified directory; false otherwise.

				Example:

					```NVGT
					void main() {
						string directory = input_box("Chdir Test", "Enter a valid path to a directory: ");
						// Could we switch to it?
						if (chdir (directory))
							alert("Success", "That directory is valid and could be switch to.");
						else 
							alert("Failure", "The directory could not be switched to. Check that you have typed the name correctly and that the directory exists.");
					}



				cwdir:

					Check the current working directory.

					string cwdir();

				Returns:

					string: a string containing the current working directory.

				Example:

					```NVGT
					void main() {
						alert("Current Working Directory is", cwdir());
					}



				environment_variable_exists:

					Check if the given environment variable exists.

					bool environment_variable_exists(const string&in name);

				Arguments:

					* const string&in name: the name of the environment variable to check.

				Returns:

					bool: true if the specified environment variable exists, false otherwise.

				Example:

					```NVGT
					void main() {
						bool exists = environment_variable_exists("PATH");
						if (exists)
							alert("Info", "The PATH variable exists.");
						else
							alert("Info", "The PATH variable does not exist.");
					}



				expand_environment_variables:

					Expand environment variables.

					string expand_environment_variables(const string&in variables);

				Arguments:

					* const string&in variables: A string containing the variable(s) you wish to expand.

				Returns:

					string: the expanded environment variable(s) if successful or an empty string if not.

				Example:

					```NVGT
					void main() {
						// The end goal here is to obtain the user's home directory on the running system if possible. This logic happens in a below function; here we just display the result.
						alert("Result", get_home_directory());
					}
					string get_home_directory() {
						if (system_is_windows) return expand_environment_variables("%USERPROFILE%");
						else if (system_is_unix) return expand_environment_variables("$HOME");
						else return "Unknown";
					}



				get_preferred_locales:

					Get a list of the user's preferred locales.

					string[]@ get_preferred_locales();

				Returns:

					string[]@: a handle to an array containing the locale names (as strings).

				Example:

					```NVGT
					void main() {
						string[]@ locales = get_preferred_locales();
						string result; // To be shown to the user.
						for (uint i = 0; i < locales.length(); i++)
							result += locales[i] + ",\n";
						// Strip off the trailing comma and new line.
						result.trim_whitespace_right_this();
						result.erase(result.length() - 1);
						alert("Info", "The locales on your system are: " + result);
					}



				system_power_info:

					Determine the state of the system's battery.

					system_power_state system_power_info(int&out seconds = void, int&out percent = void);

				Arguments:

					* int&out seconds = void: Where to store the number of remaining charge time in seconds.

					* int&out percent = void: Where to store the percentage of battery charge remaining.

				Returns:

					system_power_state: One of the system_power_state constants.

				Remarks:

					This could be useful for various reasons, such as backing up game data on low battery or showing extra mercy to online game players who run out of battery power who you might otherwise punish/tag for unlawful disconnection.

					If the battery percentage or time remaining cannot be determined, that value will be set to -1.

					Whether a time or percentage is available is up to the underlying operating system.

				Example:

					```NVGT
					// Utility function to convert a system_power_state to a string.
					string describe_power_state(system_power_state st) {
						switch (st) {
							case POWER_STATE_ERROR: return "error";
							case POWER_STATE_ON_BATTERY: return "on battery";
							case POWER_STATE_NO_BATTERY: return "no battery";
							case POWER_STATE_CHARGING: return "charging";
							case POWER_STATE_CHARGED: return "charged";
							default: return "unknown";
						}
					}
					void main() {
						system_power_state old_state = system_power_info();
						show_window(describe_power_state(old_state));
						while (!key_pressed(KEY_ESCAPE) and !key_pressed(KEY_AC_BACK)) {
							wait(5);
							int seconds, percent;
							system_power_state st = system_power_info(seconds, percent);
							if (st != old_state) show_window(describe_power_state(st));
							old_state = st;
							if (key_pressed(KEY_SPACE)) {
								string time_str = seconds > -1? " (" + timespan(seconds, 0).format() + ")" : "";
								screen_reader_speak(percent > -1? percent + "%" + time_str : "unable to determine battery percent");
							}
						}
					}



				write_environment_variable:

					Write to an environment variable.

					void write_environment_variable(const string&in variable, const string&in value);

				Arguments:

					* const string&in variable: The environment variable you wish to write to.

					* const string&in value: The value you wish to write to this variable.

				Example:

					```NVGT
					void main() {
						write_environment_variable("NVGT_Test_for_docs", "Testing");
						alert("Result", read_environment_variable("NVGT_Test_for_docs"));
					}



			Global Properties:
				COMMAND_LINE:

					COMMAND_LINE returns anything that is passed as command line arguments. Note that it returns anything after the application name, and keep in mind that you will have to parse the output yourself (basic example below).

					const string COMMAND_LINE;

				Example:

					```NVGT
					void main() {
						const string[]@ arguments = COMMAND_LINE.split(" ");
						// Did we get any arguments?
						if (arguments[0] == "")
							alert("Command Line", "No arguments provided.");
						else
							alert("Command Line", "The first argument is " + arguments[0]);
					}



				PLATFORM:

					This property returns a string containing the current platform, such as "Windows NT".

					const string PLATFORM;

				Example:

					```NVGT
					void main() {
						alert("Your current Platform  is", PLATFORM);
					}



				PLATFORM_ARCHITECTURE:

					This property returns a string containing the current platform architecture, such as "AMD64".

					const string PLATFORM_ARCHITECTURE;

				Example:

					```NVGT
					void main() {
						alert("Your current Platform architecture is", PLATFORM_ARCHITECTURE);
					}



				PLATFORM_DISPLAY_NAME:

					This property returns a string containing the current platform display name, such as "Windows 8".

					const string PLATFORM_DISPLAY_NAME;

				Remarks:

					Due to a backwards compatibility problem in windows, this function by default will cap out at Windows 8 even if the user is running a newer version of Windows. If determining the most modern windows version on the user's system is important to you, you can create a gamename.exe.manifest file to target your app for modern windows. Here is some [microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/sysinfo/targeting-your-application-at-windows-8-1) that explains how, you can probably just use the example manifest provided there.

				Example:

					```NVGT
					void main() {
						alert("Your current Platform display name is", PLATFORM_DISPLAY_NAME);
					}



				PLATFORM_VERSION:

					This property returns a string containing the current platform version, such as "6.2 (Build 9200)".

					const string PLATFORM_VERSION;

				Example:

					```NVGT
					void main() {
						alert("Your current Platform version is", PLATFORM_VERSION);
					}



				PROCESSOR_COUNT:

					This property returns the number of processors on your system. Note that this returns the number of threads, not physical cores.

					const uint PROCESSOR_COUNT;

				Example:

					```NVGT
					void main() {
						alert("Processor Threads Count", PROCESSOR_COUNT);
					}



				system_is_chromebook:

					This property is true if the application is running on a Google Chromebook.

					const bool system_is_chromebook;

				Example:

					```NVGT
					void main() {
						if(system_is_chromebook)
							alert("example", "This application is running on a chromebook!");
						else
							alert("example", "This application is not running on a chromebook.");
					}



				system_is_DeX_mode:

					This property is true if the application is running in Samsung DeX mode, which is a desktop-like environment available on certain Samsung devices. It allows a mobile device to provide a desktop experience when connected to an external display.

					const bool system_is_DeX_mode;

				Example:

					```NVGT
					void main() {
						if(system_is_DeX_mode)
							alert("example", "This application is running in Samsung DeX mode!");
						else
							alert("example", "This application is not running in Samsung DeX mode.");
					}



				system_is_mobile:

					This property is true if the application is running on any mobile device, for example a cell phone.

					const bool system_is_mobile;

				Example:

					```NVGT
					void main() {
						if(system_is_mobile)
							alert("example", "This application is running on a mobile device!");
						else
							alert("example", "This application is not running on a mobile device.");
					}



				system_is_tablet:

					This property is true if the application is running on any device that identifies itself as a tablet.

					const bool system_is_tablet;

				Example:

					```NVGT
					void main() {
						if(system_is_tablet)
							alert("example", "This application is running on a tablet!");
						else
							alert("example", "This application is not running on a tablet.");
					}



				system_is_unix:

					This property is true if the application is running on a unix operating system, which in regards to NVGT's platforms is almost everything other than windows.

					const bool system_is_unix;

				Example:

					```NVGT
					void main() {
						if(system_is_unix)
							alert("example", "This application is running on a unix operating system!");
						else
							alert("example", "This application is probably running on windows, or in any case not a unix operating system");
					}



				system_is_windows:

					This property is true if the application is running on a windows operating system.

					const bool system_is_windows;

				Example:

					```NVGT
					void main() {
						if(system_is_windows)
							alert("example", "This application is running on windows!");
						else
							alert("example", "This application is not running on windows.");
					}



				system_node_id:

					This property returns the node ID, or another words, primary MAC address, of your system.

					const string system_node_id;

				Example:

					```NVGT
					void main() {
						alert("Your system node/host ID is", system_node_id);
					}



				system_node_name:

					This property returns the node name, or another word, host name, of your system.

					const string system_node_name;

				Example:

					```NVGT
					void main() {
						alert("Your system node/host name is", system_node_name);
					}



		Filesystem:
			This section contains documentation of functions and properties that allow you to check the existance of / delete a file, create / enumerate directories, and in other ways interact with and/or manipulate the filesystem.

			If you wish to specifically read and write files, you should look at the file datastream class.


			Functions:
				directory_create:

					Creates a directory if it doesn't already exist.

					bool directory_create(const string&in directory);

				Arguments:

					* const string&in directory: path to the directory to create (can be nested, relative or absolute).

				Returns:

					bool: true if the directory was successfully created or already exists, false otherwise.

				Example:

					```NVGT
					void main() {
						if (directory_exists("test")) {
							alert("Info", "The test directory already exists; nothing to do");
							exit();
						}
						if (directory_create("test")) {
							alert("Info", "Directory created. Deleting...");
							alert("Info", directory_delete("test") ? "Success": "Fail");
						}
						else
							alert("Error", "Couldn't create the directory.");
					}



				directory_delete:
					Deletes a directory.

					`bool directory_delete(string directory);`

				Arguments:
					* string directory: the directory to delete.

				Returns:
					bool: true if the directory was successfully deleted, false otherwise.


				directory_exists:

					Query whether or not a directory exists.

					bool directory_exists(const string&in directory);

				Arguments:

					* const string&in directory: the directory whose existence will be checked (can be a relative or absolute path).

				Returns:

					bool: true if the directory exists, false otherwise.

				Example:

					```NVGT
					void main() {
						alert("Directory exists", directory_exists("test") ? "There is a test folder in the current directory" : "There is not a test folder in the current directory");
					}



				file_copy:
					Coppies a file from one location to another.

					`bool file_copy(const string&in source_filename, const string&in destination_filename);`

				Arguments:
					* const string&in source_filename: the path/name of the file to be coppied.
					* const string&in destination_filename: the path (including filename) to copy the file to.

				Returns:
					bool: true if the file was successfully coppied, false otherwise.


				file_delete:
					Deletes a file.

					`bool file_delete(const string&in filename);`

				Arguments:
					* const string&in filename: the name of the file to delete.

				Returns:
					bool: true if the file was successfully deleted, false otherwise.


				file_exists:
					Check for the existence of a particular file.

					`bool file_exists(const string&in file_path);`

				Arguments:
					* const string&in file_path: the path to the file to query.

				Returns:
					bool: true if the file exists, false otherwise.


				file_get_contents:

					Reads the entire content of a file and returns it as a string.

					string file_get_contents(string filename)

				Arguments:

					* string filename: the name of the file to read.

				Returns:

					The contents of the file on success, an empty string on failure.

				Remarks:

					This is a convenience function, though it is currently the only way to read an Android asset. If you wish to access a file with more control, see the file datastream class.

					Be careful to avoid reading files that are too large with this function, as the entire file will be in memory in order to return it as a continguous string.

				Example:

					```NVGT
					void main() {
						string filename = input_box("Filename", "Enter the name of a file to read.", "");
						string contents = file_get_contents(filename);
						if (contents == "")
							alert("Example", "Either the file was not found, or it contained no text.");
						else {
							clipboard_set_text(contents);
							alert("Example", "The contents of the file is now on your clipboard.");
						}
					}



				file_get_date_created:

					Get a datetime object representing the creation date of a particular file.

					datetime file_get_date_created(const string&in filename);

				Arguments:

					* const string&in filename; the name of the file to check.

				Returns:

					datetime: an initialized datetime object, with all its fields set to the creation date of the file.

				Remarks:

					The date returned is in UTC, not your local timezone.

				Example:

					```NVGT
					void main() {
						datetime dt = file_get_date_created("file_get_date_created.nvgt");
						alert("file_get_date_created.nvgt was created on", dt.year + "-" + dt.month + "-" + dt.day + ", " + dt.hour +":" + dt.minute + ":" + dt.second);
					}



				file_get_date_modified:

					Get a datetime object representing the modification date of a particular file.

					datetime file_get_date_modified(const string&in filename);

				Arguments:

					* const string&in filename: the name of the file to check.

				Returns:

					datetime: an initialized datetime object, with all its fields set to the modification date of the file.

				Remarks:

					The date returned is in UTC, not your local timezone.

				Example:

					```NVGT
					void main() {
						datetime dt = file_get_date_modified("file_get_date_modified.nvgt");
						alert("file_get_date_modified.nvgt was modified on", dt.year + "-" + dt.month + "-" + dt.day + ", " + dt.hour +":" + dt.minute + ":" + dt.second);
					}



				file_get_size:

					Get the size of a file (in bytes).

					int64 file_get_size(const string&in filename);

				Arguments:

					* const string&in filename: the name/path of the file to get the size of.

				Returns:

					int64: the size of the file (in bytes).

				Example:

					```NVGT
					void main() {
						alert("file_get.size.nvgt is", file_get_size("file_get_size.nvgt") + " bytes");
					}



				file_put_contents:

					Writes a string to a file.

					bool file_put_contents(string filename, string content, bool append = false)

				Arguments:

					* string filename: the name of the file to write to.

					* string content: the content to write.

					* bool append = false: specifies whether the current contents of the file should be overwritten when writing.

				Returns:

					bool: true on success, false on failure.

				Example:

					```NVGT
					void main() {
						if (!file_put_contents("example.txt", "This is an example"))
							alert("Example", "Failed to write the file.");
						else
							alert("Example", "Successfully wrote the example file.");
					}



				get_preferences_path:

					Determines the best location to store user data for your game, creating the directory for you if necessary.

					string get_preferences_path(string company_name, string product_name);

				Arguments:

					* string company_name: The name of your organization, can be blank.

					* string product_name: The name of your game or application.

				Remarks:

					This is the recommended way to determine where your game should store application data. While other properties exist such as DIRECTORY_APPDATA which can be used to compose your own path if you desire, this function is generally more robust and platform agnostic than trying to prepare your own storage path using directory constants. This also frees you from needing to check for the existance of the directory before writing to it.

					The company name and product name should never change for an application once you start using them, lest players lose access to their previous save data due to the directory changing names. The company name can be blank if you wish to only create one level of new directories in the users appdata, though this is generally not recommended to avoid clutter.

					Both the company and product name are expected to contain only spaces and alphanumeric characters if you wish to be as platform agnostic as possible.

					You should not assume that DIRECTORY_APPDATA + "/" + company_name + "/" + product_name will be the same as the directory returned by this function, or even that both the company and product names will be used all the time. On android, for example, this function always just returns the app's internal storage path.

					This function is also registered as the alias DIRECTORY_PREFERENCES, for those who prefer the old style of directory constants naming.

				Example:

					```NVGT
					void main() {
						string path = get_preferences_path("", "documentation example");
						alert("test", "data will be stored at " + path);
						directory_delete(path); // In this example we don't actually want this directory to save.
					}



				glob:

					Return a list of files and directories on the filesystem given a path glob.

					string[]@ glob(const string&in path_pattern, glob_options options = GLOB_DEFAULT);

				Arguments:

					* const string&in path_pattern: The pattern to match files and directories with (Unix shell like, see remarks).

					* glob_options options: A bitwise of glob_options enum constants that influence the behavior of this function (see remarks).

				Returns:

					string[]@: A list of all matching files and directories that match the given path pattern, an empty array on no matches or failure.

				Remarks:

					This function provides for one of the easiest ways to enumerate the filesystem in NVGT, particularly because the path patterns provided can actually cause semi-recursive directory searches. The search starts at the current working directory unless an absolute path is given.

					The glob patterns have simple rules:

					* path separators must be matched exactly, \* will not cause a recursive lookup

					* \* matches any sequence of characters

					* ? matches any single character

					* [set] matches any characters between the brackets

					* [!set] matches any characters that are not listed between the brackets

					* `\*, \[, \] etc` exactly match a special character usually used as part of the glob expression

					There is no guarantee that the items returned will appear in any particular order in the array.

					The following glob_options constance are defined:

					* GLOB_DEFAULT: the default options

					* GLOB_IGNORE_HIDDEN: do not match when directory entries begin with a .

					* GLOB_FOLLOW_SYMLINKS: traverse even across symbolic links if the given pattern demands it

					* GLOB_CASELESS: match case insensitively

				Example:

					```NVGT
					void main() {
						// List all .nvgt files within the filesystem documentation directory.
						string[]@ items = glob("../*/*.nvgt");
						alert("files found", string(items));
					}



			Global Properties:
				DIRECTORY_APPDATA:

					Property that returns the user's roaming application directory, which is usually where game data can be written to.

					const string DIRECTORY_APPDATA;

				remarks:

					A slash character is already appended to the directory returned by this property.

					This function may return different values depending on the operating system the application is being run on.

					* On Windows, usually C:\Users\%username%\appdata\roaming/.

					* on macOS, usually ~/Library/Preferences/.

					* on Linux, usually ~/.config/.

					* on Android, the app's internal storage path.

					In any case, the directory returned should be writable.

				Example:

					```NVGT
					void main() {
						alert("example", "data for the game could be stored at " + DIRECTORY_APPDATA + "my_game/");
					}



				DIRECTORY_TEMP:

					Property that returns the system's temporary directory, a place where short-term data can be stored.

					const string DIRECTORY_TEMP;

				remarks:

					A slash character is already appended to the directory returned by this property.

					This function may return different values depending on the operating system the application is being run on.

					* On Windows, usually C:\Users\%username%\appdata\local\temp/.

					* on macOS and Linux systems, usually /tmp/.

					* on Android, the app's cache path.

					In any case the directory returned should be writable, though you should expect things you store there to be deleted by external factors at any time.

				Example:

					```NVGT
					void main() {
						alert("example", "a temporary file for the game could be stored at " + DIRECTORY_TEMP + "filename.txt");
					}



		Math:
			Classes:
				aabb:

					A class representing an axis-aligned bounding box (AABB).

					aabb();

				Remarks:

					AABBs are numbers representing the corners of a fixed, non-rotating, rectangular box. This is useful for such functions as collision detection, defining world regions and so on.

				Example:

					```NVGT
					void main() {
						aabb box1;
						aabb box2;
						box1.min = vector(1, 1, 1);
						box1.max = vector(5, 5, 5);
						box2.min = vector(2, 2, 2);
						box2.max = vector(4, 4, 4);
					if (box1.contains(box2)) alert("Boxed in a box!", "Box1 contains box2.");
					}



					Methods:
						apply_scale:

							Scales the box by multiplying both min and max directly with the given scale vector. This causes the box to scale relative to the origin (0,0,0), meaning both its size and position will change if it's not centered at the origin.

							void aabb::apply_scale(const vector&in scale);

						Arguments:

							* scale: A vector containing the factors to scale by.

						Example:

							```NVGT
							void main() {
								aabb b;
								b.min = vector(1, 3, 5);
								b.max = vector(3, 5, 7);
								b.apply_scale(vector(2, 2, 2));
								alert("The bottom left corner of the box is located at", round(b.min.x, 0)+", "+round(b.min.y, 0)+", "+round(b.min.z, 0)); // 2, 6, 10
								alert("The top right corner of the box is located at", round(b.max.x, 0)+", "+round(b.max.y, 0)+", "+round(b.max.z, 0)); // 6, 10, 14
							}



						contains:

							Checks whether another AABB is fully contained within this one.

							bool aabb::contains(const aabb&in other) const;

							Checks whether the given point lies inside this AABB, optionally with a margin of error (epsilon).

							bool contains(const vector&in point, float epsilon = EPSILON) const

						Arguments (1):

							* other: Another AABB to compare.

						Arguments (2):

							* point: The point to check for.

							* epsilon: Error tolerance level (default is constant EPSILON, which should be fine for most cases).

						Remarks (1):

							Unlike the test_collision method, this method tests whether one box is fully contained inside another. If any part of the inner box is outside the boundaries of the first, this method will return false.

						Remarks (2):

							EPSILON is a very small floating point constant (1.19e-7) used to allow tolerance in equality or boundary checks due to rounding errors in floating point math. Changing this to a smaller value (such as 0) could lead to subtle bugs when comparing floating point values, while changing it to a larger value could provide incorrect results. It is therefore recommended to leave this parameter untouched unless you are changing it for a specific edge case.

						Example:

							```NVGT
							void main() {
								aabb b1;
								aabb b2;
								b1.min = vector(1, 1, 1);
								b1.max = vector(5, 5, 5);
								b2.min = vector(3, 3, 3);
								b2.max = vector(5, 5, 5);
								// Both these conditions should be true.
								if (b1.contains(b2)) alert("Boxed in a box!", "box1 contains box2.");
								if (b1.contains(vector(1,2,3))) alert("Point found!", "box1 contains point 1,2,3.");
							}



						inflate:

							Expands the box by the given amounts in all directions. This means that the total size increases by twice the given values.

							void aabb::inflate(float x, float y, float z);

						Arguments:

							* x: Value to inflate X axis.

							* y: Value to inflate Y axis.

							* z: Value to inflate Z axis.

						Example:

							```NVGT
							void main() {
								aabb b;
								b.min = vector(1, 3, 5);
								b.max = vector(2, 4, 6);
								b.inflate(1, 1, 1);
								alert("The bottom left corner of the box is located at", round(b.min.x, 0)+", "+round(b.min.y, 0)+", "+round(b.min.z, 0));
								alert("The top right corner of the box is located at", round(b.max.x, 0)+", "+round(b.max.y, 0)+", "+round(b.max.z, 0));
							}



						inflate_with_point:

							Expands the box only if necessary to ensure the given point is contained within it. If the point is already inside, nothing changes.

							void aabb::inflate_with_point(const vector&in point);

						Arguments:

							point: The point to include.

						Example:

							```NVGT
							void main() {
								aabb b;
								b.min = vector(1, 3, 5);
								b.max = vector(3, 5, 7);
								b.inflate_with_point(vector(3, 6, 9)); // Only the max should change here.
								alert("The bottom left corner of the box is located at", round(b.min.x, 0)+", "+round(b.min.y, 0)+", "+round(b.min.z, 0));
								alert("The top right corner of the box is located at", round(b.max.x, 0)+", "+round(b.max.y, 0)+", "+round(b.max.z, 0));
							}



						merge:

							Sets this AABB to be the bounding box that contains both aabb1 and aabb2.

							void aabb::merge(const aabb&in aabb1, const aabb&in aabb2);

						Arguments:

							* aabb1: The first box to merge.

							* aabb2: The second box to merge.

						Example:

							```NVGT
							void main() {
								aabb b1;
								aabb b2;
								aabb b3;
								b1.min = vector(0, 0, 0);
								b1.max = vector(50, 500, 5000);
								b2.min = vector(1, 3, 5);
								b2.max = vector(5, 5, 5);
								b3.min = vector(9, 9, 9);
								b3.max = vector(12, 15, 18);
								b1.merge(b2, b3); // Completely changes the size to merge 2 and 3 into 1.
								alert("The bottom left corner of the box is located at", round(b1.min.x, 0)+", "+round(b1.min.y, 0)+", "+round(b1.min.z, 0));
								alert("The top right corner of the box is located at", round(b1.max.x, 0)+", "+round(b1.max.y, 0)+", "+round(b1.max.z, 0));
							}



						merge_with:

							Expands this AABB to fully contain both itself and the other AABB.

							void aabb::merge_with(const aabb&in aabb);

						Arguments:

							aabb: The AABB to merge.

						Example:

							```NVGT
							void main() {
								aabb b1;
								aabb b2;
								b1.min = vector(1, 3, 5);
								b1.max = vector(5, 5, 5);
								b2.min = vector(9, 9, 9);
								b2.max = vector(12, 15, 18);
								b1.merge_with(b2);
								alert("The bottom left corner of the box is located at", round(b1.min.x, 0)+", "+round(b1.min.y, 0)+", "+round(b1.min.z, 0));
								alert("The top right corner of the box is located at", round(b1.max.x, 0)+", "+round(b1.max.y, 0)+", "+round(b1.max.z, 0));
							}



						raycast:

							Casts a ray at the box and, if it intersects, gives you the exact point where it hits.

							bool aabb::raycast(const ray&in ray, vector&out hit_point);

						Arguments:

							* ray: The ray to cast.

							* hit_point (out): Stores the hit point where the ray collided with the box.

						Example:

							```NVGT
							void main() {
								aabb b;
								b.min = vector(1, 1, 1);
								b.max = vector(10, 10, 2);
								vector start = vector(50, 50, 1);
								vector target = vector(5, 5, 1);
								ray r(start, target, 100);
								vector hit;
								if (b.raycast(r, hit)) alert("Boom!", "We collided at "+round(hit.x, 0)+", "+round(hit.y, 0)+", "+round(hit.z, 0)+".");
							}



						test_collision:

							Checks if the position of one box would collide or overlap with another.

							bool aabb::test_collision(const aabb&in aabb) const;

						Arguments:

							aabb: The box to compare with.

						Remarks:

							Where the contains() method checks whether one box is fully contained within another, this method merely checks whether two boxes are touching.

						Example:

							```NVGT
							void main() {
								aabb b1;
								aabb b2;
								b1.min = vector(5, 5, 0);
								b1.max = vector(10, 10, 0);
								b2.min = vector(1, 1, 0);
								b2.max = vector(5, 5, 0);
								if (b1.test_collision(b2)) alert("Bang!", "We collided.");
							}



						test_collision_triangle_aabb:

							Checks whether a triangle (given by 3 points) intersects the AABB.

							bool aabb::test_collision_triangle_aabb(const array<vector>@ points) const;

						Arguments:

							points: The points of the triangle.

						Remarks:

							This method would be the same as placing the minimum and maximum vectors of the triangle inside another AABB and calling test_collision.

							Note that if the input parameter doesn't represent a triangle (I.E. have exactly three points), the method will throw an exception.

						Example:

							```NVGT
							void main() {
								vector[] points;
								points.insert_last(vector(2, 2, 0));
								points.insert_last(vector(8, 2, 0));
								points.insert_last(vector(5, 5, 0));
								aabb b;
								b.min = vector(1, 1, 0);
								b.max = vector(12, 15, 5);
								if (b.test_collision_triangle_aabb(points)) alert("Bang!", "We collided.");
							}



						test_ray_intersect:

							Checks whether a ray (straight line) intersects the box, given a starting point, direction, and distance.

							bool aabb::test_ray_intersect(const vector&in ray_origin, const vector&in ray_direction, float ray_max_fraction);

						Arguments:

							* ray_origin: The starting position.

							* ray_direction: The direction you expect to move.

							* ray_max_fraction: The maximum distance you expect to travel.

						Remarks:

							This is useful for weapon aiming or pathfinding systems.

						Example:

							```NVGT
							void main() {
								aabb b;
								b.min = vector(1, 1, 1);
								b.max = vector(10, 10, 2);
								vector start = vector(50, 50, 1);
								vector target = vector(5, 5, 1);
								vector dir = (target - start);
								dir.normalize();
								if (b.test_ray_intersect(start, dir, 100)) alert("Boom!", "We collided.");
							}



					Properties:
						center:

							A read-only property representing the center of the box, calculated based on min and max.

							vector aabb::center const;

						Example:

							```NVGT
							void main() {
								aabb b;
								b.min = vector(1, 3, 5);
								b.max = vector(5, 3, 7);
								alert("The center of the box is located at", round(b.center.x, 0)+", "+round(b.center.y, 0)+", "+round(b.center.z, 0));
							}



						extent:

							A read-only property representing the extent of the box, calculated based on min and max.

							Note that "extent" calculates from min, not from center. Therefore in this case, extent is synonymous with size.

							vector aabb::extent const;

						Example:

							```NVGT
							void main() {
								aabb b;
								b.min = vector(1, 3, 5);
								b.max = vector(5, 3, 7);
								alert("The extent of the box is located at", round(b.extent.x, 0)+", "+round(b.extent.y, 0)+", "+round(b.extent.z, 0)); // 4,0,2.
							}



						max:

							Represents the maximum coordinates (top right corner) of the box.

							vector aabb::max const;

						Example:

							```NVGT
							void main() {
								aabb b;
								b.max = vector(5, 3, 7);
								alert("The top right corner of the box is located at", round(b.max.x, 0)+", "+round(b.max.y, 0)+", "+round(b.max.z, 0));
							}



						min:

							Represents the minimum coordinates (bottom left corner) of the box.

							vector aabb::min const;

						Example:

							```NVGT
							void main() {
								aabb b;
								b.min = vector(1, 3, 5);
								alert("The bottom left corner of the box is located at", round(b.min.x, 0)+", "+round(b.min.y, 0)+", "+round(b.min.z, 0));
							}



						volume:

							A read-only property representing the volume of the box, calculated based on min and max.

							Note that "volume" here refers to total number of points, and is defined as extent.x * extent.y * extent.z.

							float aabb::volume const;

						Example:

							```NVGT
							void main() {
								aabb b;
								b.min = vector(1, 2, 3);
								b.max = vector(2, 4, 6);
								alert("The volume of the box is", b.volume); // 6.
							}



				complex:

					A class representing a complex number, composed of a real part `r` and an imaginary part `i`.

					1. complex();

					2. complex(const complex&in other);

					3. complex(float r, float i = 0.0);

				Arguments (3):

					* float r: The real part of the complex number.

					* float i = 0.0: The imaginary part of the complex number. Defaults to 0 if not specified.

				Remarks:

					The complex class supports basic arithmetic operations (add, subtract, multiply, divide) and can be compared for equality. It also provides access to magnitude (`abs()`), and utilities for working with real/imaginary order via `ri` and `ir` properties.

				Example:

					```NVGT
					void main() {
						complex a(2, 3); // 2 + 3i
						complex b(1, -1); // 1 - i
						complex c = a + b; // (2+1, 3-1) = (3, 2)
						alert("result", "real: " + c.r + ", imag: " + c.i);
					}



					Functions:
						abs:

							Returns the absolute value (magnitude) of the complex number.

							float complex::abs() const;

						Example:

							```NVGT
							void main() {
								complex c(3, 4);
								alert("Magnitude =", c.abs());
							}



					Operators:
						opAdd:

							Adds two complex numbers and returns the result.

							complex complex::opAdd(const complex&in other) const;

						Arguments:

							* const complex&in other: the complex number to add.

						Returns:

							complex: a new complex object containing the value of the addition.

						Example:

							```NVGT
							// Helper function to print a complex number.
							string to_string(const complex&in c) {
								return "(" + c.r + ", " + c.i + ")";
							}
							void main() {
								complex a(1, 2);
								complex b(3, 4);
								complex c = a + b;
								alert("a + b =", to_string(c));
							}



						opAddAssign:

							Adds another complex number to this one.

							complex& complex::opAddAssign(const complex&in other);

						Arguments:

							* const complex&in other: the complex number to add.

						Returns:

							complex&: a reference to the complex number being operated on.

						Example:

							```NVGT
							// Helper function to print a complex number.
							string to_string(const complex&in c) {
								return "(" + c.r + ", " + c.i + ")";
							}
							void main() {
								complex a(1, 2);
								complex b(3, 4);
								a += b;
								alert("a + b =", to_string(a));
							}



						opDiv:

							Divides this complex number by another and returns the result.

							complex complex::opDiv(const complex&in other) const;

						Arguments:

							* const complex&in other: the complex number to divide by.

						Returns:

							complex: a new complex object containing the value of the division.

						Example:

							```NVGT
							// Helper function to print a complex number.
							string to_string(const complex&in c) {
								return "(" + c.r + ", " + c.i + ")";
							}
							void main() {
								complex a(1, 2);
								complex b(3, 4);
								complex c = a / b;
								alert("a / b =", to_string(c));
							}



						opDivAssign:

							Divides this complex number by another.

							complex& complex::opDivAssign(const complex&in other);

						Arguments:

							* const complex&in other: the complex number to divide by.

						Returns:

							complex&: a reference to the complex number being operated on.

						Example:

							```NVGT
							// Helper function to print a complex number.
							string to_string(const complex&in c) {
								return "(" + c.r + ", " + c.i + ")";
							}
							void main() {
								complex a(1, 2);
								complex b(3, 4);
								a /= b;
								alert("a / b =", to_string(a));
							}



						opEquals:

							Checks if two complex numbers are equal.

							bool complex::opEquals(const complex&in other) const;

						Arguments:

							* const complex&in other: the complex number to compare to.

						Returns:

							bool: true if the two complex numbers are equal, false otherwise.

						Example:

							```NVGT
							void main() {
								complex a(1, 2);
								complex b(1, 2);
								alert("a equals b?", a == b);
							}



						opMul:

							Multiplies two complex numbers and returns the result.

							complex complex::opMul(const complex&in other) const;

						Arguments:

							* const complex&in other: the complex number to multiply with.

						Returns:

							complex: a new complex object containing the value of the multiplication.

						Example:

							```NVGT
							// Helper function to print a complex number.
							string to_string(const complex&in c) {
								return "(" + c.r + ", " + c.i + ")";
							}
							void main() {
								complex a(1, 2);
								complex b(3, 4);
								complex c = a * b;
								alert("a * b =", to_string(c));
							}



						opMulAssign:

							Multiplies this complex number by another.

							complex& complex::opMulAssign(const complex&in other);

						Arguments:

							* const complex&in other: the complex number to multiply with.

						Returns:

							complex&: a reference to the complex number being operated on.

						Example:

							```NVGT
							// Helper function to print a complex number.
							string to_string(const complex&in c) {
								return "(" + c.r + ", " + c.i + ")";
							}
							void main() {
								complex a(1, 2);
								complex b(3, 4);
								a *= b;
								alert("a * b =", to_string(a));
							}



						opSub:

							Subtracts a complex number from this one and returns the result.

							complex complex::opSub(const complex&in other) const;

						Arguments:

							* const complex&in other: the complex number to subtract.

						Returns:

							complex: a new complex object containing the value of the subtraction.

						Example:

							```NVGT
							// Helper function to print a complex number.
							string to_string(const complex&in c) {
								return "(" + c.r + ", " + c.i + ")";
							}
							void main() {
								complex a(5, 6);
								complex b(2, 3);
								complex c = a - b;
								alert("a - b =", to_string(c));
							}



						opSubAssign:

							Subtracts another complex number from this one.

							complex& complex::opSubAssign(const complex&in other);

						Arguments:

							* const complex&in other: the complex number to subtract.

						Returns:

							complex&: a reference to the complex number being operated on.

						Example:

							```NVGT
							// Helper function to print a complex number.
							string to_string(const complex&in c) {
								return "(" + c.r + ", " + c.i + ")";
							}
							void main() {
								complex a(5, 6);
								complex b(2, 3);
								a -= b;
								alert("a - b =", to_string(a));
							}



					Properties:
						i:

							Represents the imaginary component of the complex number.

							float complex::i;

						Example:

							```NVGT
							void main() {
								complex c(0, 1);
								alert("The imaginary part is", c.i);
							}



						ir:

							Accesses the complex number in imaginary-real order.

							complex complex::ir [property];

						Example:

							```NVGT
							// Helper function to print a complex number.
							string to_string(const complex&in c) {
								return "(" + c.r + ", " + c.i + ")";
							}
							void main() {
								complex c(1, 2);
								alert("Imaginary-Real =", to_string(c.ir));
							}



						ri:

							Accesses the complex number in real-imaginary order.

							complex complex::ri;

						Example:

							```NVGT
							// Helper function to print a complex number.
							string to_string(const complex&in c) {
								return "(" + c.r + ", " + c.i + ")";
							}
							void main() {
								complex c(1, 2);
								alert("Real-Imaginary =", to_string(c.ri));
							}



				vector:

					A class containing x, y and z coordinates, usually used to represent a 3d point in space.

					1. vector();

					2. vector(const vector& in vec);

					3. vector(float x, float y = 0.0, float z = 0.0);

				Arguments (3):

					* float x: The initial x point or coordinate this vector represents.

					* float y = 0.0: The initial y point or coordinate this vector represents.

					* float z = 0.0: The initial z point or coordinate this vector represents, defaulted to 0 because it may not be needed in some 2d applications.

				Remarks:

					An advantage with vectors is that they contain basic addition and scaling operators.

				Example:

					```NVGT
					void main() {
						vector v1(5, 5, 5);
						vector v2(10, 10, 0);
						vector v3 = v1 + v2;
						alert("example", "the new vector is "+v3.x + ", " + v3.y + ", " + v3.z); // will show that the vector is 15, 15, 5.
					}



					Methods:
						cross:

							Returns the cross product of two vectors.

							vector vector::cross(const vector&in other);

						Arguments:

							* other: The second vector to calculate from.

						Remarks:

							The cross product provides a third vector which is Perpendicular to (at right-angles with) the first two vectors. This is useful for working out relative directions (such as forwards, right, up) based on the world's orientation.

						Example:

							```NVGT
							void main() {
								vector v1(0, 1, 0); // Forwards
								vector v2(0, 0, 1); // Up
								vector v3=v1.cross(v2); // right
								alert("Cross", v3.x+", "+v3.y+", "+v3.z);
							}



						dot:

							Returns the dot product of two vectors.

							float vector::dot(const vector&in other);

						Arguments:

							* other: The vector to sum.

						Remarks:

							The dot product is a measure of how much two vectors point in the same direction. This is often used in physics for checking if an object is moving into or away from a surface, measuring forces, or calculating angles.

							* Returns a positive value if both vectors point roughly in the same direction.

							* Returns zero if the vectors are perpendicular (at right angles, not pointing the same way at all).

							* Returns a negative value if the vectors point in opposite directions.

							The dot product is calculated as (ax * bx) + (ay * by) + (az * bz).

						Example:

							```NVGT
							void main() {
								vector v1(1, 2, 3);
								vector v2(4, 5, 6);
								alert("Dot", v2.dot(v1)); // (1 * 4) + (2 * 5) + (3 * 6) = 32.
							}



						length:

							Retrieves the Euclidean length of the vector.

							float vector::length();

						Arguments:

							None.

						Remarks:

							It is important to emphasize this function returns the vector's Euclidian length (also known as its magnitude). This is calculated as sqrt((x * x) + (y * y) + (z * z)), which gives the distance from the origin (0, 0, 0) to the point represented by the vector.

							While this may sound abstract at first, it's extremely useful in many common scenarios. Some examples are:

							* Measuring distance: Use the length to find out how far a vector reaches from the origin. The vector (3, 4, 0) has a length of 5.0, meaning it's 5 units away in space.

							* Calculating speed: If a vector represents velocity, the length gives you the speed (how fast something is moving), while the vector's direction shows where it's going.

							* Collision detection: To determine how close two points are, subtract one position from another to get a direction vector, then use the length to measure the distance.

							If you are looking for the total size (I.E. number of units, volume etc), you will manually need to calculate this by using x * y * z.

						Example:

							```NVGT
							void main() {
								vector v1(3, 4, 0);
								alert("Length", v1.length());
							}



						length_square:

							Retrieves the square length of the vector.

							float vector::length_square();

						Arguments:

							None.

						Remarks:

							This is similar to length except it does not perform the square root calculation.

						Example:

							```NVGT
							void main() {
								vector v1(3, 4, 0);
								alert("Length", v1.length_square());
							}



						normalize:

							Normalizes the vector so that its Euclidian length is 1.

							vector& vector::normalize();

						Arguments:

							None.

						Remarks:

							Vector normalization is useful when you need to find out the direction something is pointing rather than where it goes. This is especially useful in 3D or physics-based games, where movement and aiming often happen in arbitrary directions rather than along fixed axes.

							Warning: Calling normalize() on a zero vector (0, 0, 0) will return a vector whose values are NAN due to division by zero. Be sure to check the vector's length before normalizing.

						Example:

							```NVGT
							void main() {
								vector v1(3, 4, 0);
								v1.normalize();
								alert("Vector", "(" + v1.x + ", " + v1.y + ", " + v1.z + ")");
							}



						set:

							Set new values on a vector.

							void vector::set(float x, float y, float z);

						Arguments:

							* float x: The new x point or coordinate this vector represents.

							* float y: The new y point or coordinate this vector represents.

							* float z: The new z point or coordinate this vector represents.

						Example:

							```NVGT
							void main() {
								vector v1(5, 5, 5);
								v1.set(10, 12, 14);
								alert("example", "the new vector is "+v1.x + ", " + v1.y + ", " + v1.z); // will show that the vector is 10, 12, 14.
							}



						setToZero:

							Reset the vector to 0, 0, 0.

							void vector::setToZero();

						Arguments:

							None.

						Example:

							```NVGT
							void main() {
								vector v1(5, 5, 5);
								v1.setToZero();
								alert("example", "the new vector is "+v1.x + ", " + v1.y + ", " + v1.z); // will show that the vector is 0, 0, 0.
							}



					Operators:
						opAdd:

							Overloads the + operator, allowing you to add a vector to another.

							vector& vector::opAdd(const vector &in other);

						Arguments:

							* const vector&in other: the vector to add.

						Example:

							```NVGT
							void main() {
								vector v1(2, 3, 5);
								vector v2(5, 4, 2);
								vector v3 = v1 + v2;
								alert("The third vector =", "(" + v3.x + ", " + v3.y + ", " + v3.z + ")");
							}



						opAddAssign:

							Overloads the += operator, allowing you to add a vector to another, already existing one.

							vector& vector::opAddAssign(const vector &in other);

						Arguments:

							* const vector&in other: the vector to add.

						Example:

							```NVGT
							void main() {
								vector v1(2, 3, 5);
								vector v2(5, 4, 2);
								v1 += v2;
								alert("The first vector =", "(" + v1.x + ", " + v1.y + ", " + v1.z + ")");
							}



						opAssign:

							Overloads the = operator, allowing you to assign a vector to another, already existing one.

							vector& vector::opAssign(const vector &in other);

						Arguments:

							* const vector&in other: the vector to assign.

						Example:

							```NVGT
							void main() {
								vector v1;
								vector v2(1.2, 2.3, 9.3);
								v1 = v2;
								alert("The first vector =", "(" + round(v1.x, 2) + ", " + round(v1.y, 2) + ", " + round(v1.z, 2) + ")");
							}



						opDiv:

							Overloads the / operator, allowing you to divide two vectors, or divide one vector by a scalar.

							vector& vector::opDiv(const vector &in other);

							vector& vector::opDiv(float scalar);

						Arguments (1):

							* const vector&in other: the vector to divide by.

						Arguments (2):

							* float scalar: The scalar to divide by.

						Remarks:

							Dividing two vectors will, as you might imagine, divide each axis independently. On the other hand, dividing by a scalar will divide all axes by that scalar.

						Example:

							```NVGT
							void main() {
								vector v1(3, 12, 24);
								vector v2=v1/3;
								alert("The second vector =", "(" + v2.x + ", " + v2.y + ", " + v2.z + ")");
							}



						opDivAssign:

							Overloads the /= operator, allowing you to divide one vector by a scalar.

							vector& vector::opDivAssign(float scalar);

						Arguments:

							* float scalar: The scalar to divide by.

						Remarks:

							Dividing by a scalar will divide all axes by that scalar.

						Example:

							```NVGT
							void main() {
								vector v1(3, 12, 16);
								v1 /= 3;
								alert("The first vector =", "(" + v1.x + ", " + v1.y + ", " + v1.z + ")");
							}



						opEquals:

							Overloads the == operator, checking if two vectors are equal.

							bool vector::opEquals(const vector &in other);

						Arguments:

							* const vector&in other: the vector to compare.

						Example:

							```NVGT
							void main() {
								vector v1(2, 3, 5);
								vector v2(5, 4, 2);
								if(v1 != v2)
								alert("Compared", "Vectors are not equal.");
							}



						opImplConv:

							Allows the vector to be turned into a string.

							string vector::opImplConv() const;

						Remarks:

							Note this currently does not trim redundant decimal places from the output.

						Example:

							```NVGT
							void main() {
								vector v(2, 3, 5);
								alert("Vector", v);
							}



						opIndex:

							Overloads the [] operator, allowing you to access the axes via their indexes.

							bool vector::opIndex(int index);

						Arguments:

							* int index: the index of the axis to query.

						Remarks:

							The axis index is represent by a number: x = 0, y = 1, z = 2.

						Example:

							```NVGT
							void main() {
								vector v(2, 3, 5);
								alert("Vector", v[0]+", "+v[1]+", "+v[2]);
							}



						opMul:

							Overloads the * operator, allowing you to multiply two vectors, or multiply one vector by a scalar.

							vector& vector::opMul(const vector &in other);

							vector& vector::opMul(float scalar);

						Arguments (1):

							* const vector&in other: the vector to multiply.

						Arguments (2):

							* float scalar: The scalar to multiply by.

						Remarks:

							Multiplying two vectors will, as you might imagine, multiply each axis independently. On the other hand, multiplying by a scalar will multiply all axes by that scalar.

						Example:

							```NVGT
							void main() {
								vector v1(1, 4, 8);
								vector v2=v1*3;
								alert("The second vector =", "(" + v2.x + ", " + v2.y + ", " + v2.z + ")");
							}



						opMulAssign:

							Overloads the *= operator, allowing you to multiply one vector by a scalar.

							vector& vector::opMulAssign(float scalar);

						Arguments:

							* float scalar: The scalar to multiply by.

						Remarks:

							Multiplying by a scalar will multiply all axes by that scalar.

						Example:

							```NVGT
							void main() {
								vector v1(1, 4, 8);
								v1*=3;
								alert("The first vector =", "(" + v1.x + ", " + v1.y + ", " + v1.z + ")");
							}



						opSub:

							Overloads the - operator, allowing you to subtract a vector from another.

							vector& vector::opSub(const vector &in other);

						Arguments:

							* const vector&in other: the vector to subtract.

						Example:

							```NVGT
							void main() {
								vector v1(12, 10, 8);
								vector v2(5, 3, 1);
								vector v3 = v1 - v2;
								alert("The third vector =", "(" + v3.x + ", " + v3.y + ", " + v3.z + ")");
							}



						opSubAssign:

							Overloads the -= operator, allowing you to subtract a vector from another.

							vector& vector::opSubAssign(const vector &in other);

						Arguments:

							* const vector&in other: the vector to subtract.

						Example:

							```NVGT
							void main() {
								vector v1(12, 10, 8);
								vector v2(5, 3, 1);
								v1 -= v2;
								alert("The first vector =", "(" + v1.x + ", " + v1.y + ", " + v1.z + ")");
							}



					Properties:
						is_finite:

							Checks if a vector is considered finite (I.E. it has finite components).

							bool vector::is_finite const;

						Example:

							```NVGT
							void main() {
								vector v(5, 4, 8);
								alert("Is finite?", v.is_finite);
							}



						is_unit:

							Checks if a vector has a length of 1 (I.E. the vector is normalised).

							bool vector::is_unit const;

						Remarks:

							This method is more accurate than checking each axis for 0 directly, as it takes into account the inaccuracies that can be caused by floating point arithmetic.

						Example:

							```NVGT
							void main() {
								vector v(5, 4, 8);
								v.normalize();
								alert("Is unit?", v.is_unit);
							}



						is_zero:

							Checks if a vector is 0.

							bool vector::is_zero const;

						Remarks:

							This method is more accurate than checking each axis for 0 directly, as it takes into account the inaccuracies that can be caused by floating point arithmetic.

						Example:

							```NVGT
							void main() {
								vector v(0, 0, 0);
								alert("Is zero?", v.is_zero);
							}



						max_axis:

							Returns the index of the axis which contains the maximum value.

							int vector::max_axis const;

						Remarks:

							The axis index is represent by a number: x = 0, y = 1, z = 2.

						Example:

							```NVGT
							void main() {
								vector v(5, 7, 4);
								alert("maximum axis", v.max_axis);
							}



						max_value:

							Returns the maximum value of the vector.

							float vector::max_value const;

						Example:

							```NVGT
							void main() {
								vector v(5, 7, 4);
								alert("maximum value", v.max_value);
							}



						min_axis:

							Returns the index of the axis which contains the minimum value.

							int vector::min_axis const;

						Remarks:

							The axis index is represent by a number: x = 0, y = 1, z = 2.

						Example:

							```NVGT
							void main() {
								vector v(5, 7, 4);
								alert("Minimum axis", v.min_axis);
							}



						min_value:

							Returns the minimum value of the vector.

							float vector::min_value const;

						Example:

							```NVGT
							void main() {
								vector v(5, 7, 4);
								alert("Minimum value", v.min_value);
							}



						x:

							Represents the x coordinate of the vector.

							float vector::x const;

						Example:

							```NVGT
							void main() {
								vector v(2.9);
								alert("The x value of the vector is", round(v.x, 2));
							}



						y:

							Represents the y coordinate of the vector.

							float vector::y const;

						Example:

							```NVGT
							void main() {
								vector v(2.9, 19.2);
								alert("The y value of the vector is", round(v.y, 2));
							}



						z:

							Represents the z coordinate of the vector.

							float vector::z const;

						Example:

							```NVGT
							void main() {
								vector v(2.9, 19.2, 4.1);
								alert("The z value of the vector is", round(v.z, 2));
							}



			Functions:
				abs:

					Returns the absolute value of a value.

					double abs(double x);

				Arguments:

					* double x: The value whose absolute value is to be calculated.

				Returns:

					double: the absolute value of the given value.

				Example:

					```NVGT
					void main() {
						alert("Example", "The absolute value of -5 is " + abs(-5));
					}



				acos:

					Returns the arc cosine of a value in radians.

					double acos(double x);

				Arguments:

					* double x: The value whose arc cosine is to be calculated.

				Returns:

					double: the arc cosine of the given value.

				Example:

					```NVGT
					void main() {
					    alert("Example", "The arc cosine of 0.5 is " + acos(0.5) + " radians");
					}



				asin:

					Returns the arc sine of a value in radians.

					double asin(double x);

				Arguments:

					* double x: The value whose arc sine is to be calculated.

				Returns:

					double: the arc sine of the given value.

				Example:

					```NVGT
					void main() {
					    alert("Example", "The arc sine of 0.5 is " + asin(0.5) + " radians");
					}



				atan:

					Returns the arc tangent of a value in radians.

					double atan(double x);

				Arguments:

					* double x: The value whose arc tangent is to be calculated.

				Returns:

					double: the arc tangent of the given value.

				Example:

					```NVGT
					void main() {
					    alert("Example", "The arc tangent of 1 is " + atan(1) + " radians");
					}



				atan2:

					Returns the arc tangent of y/x, where y and x are the coordinates of a point.

					double atan2(double y, double x);

				Arguments:

					* double y: The ordinate coordinate.

					* double x: The abscissa coordinate.

				Returns:

					double: the arc tangent of y/x.

				Example:

					```NVGT
					void main() {
					    alert("Example", "The arc tangent of (1, 2) is " + atan2(1, 2) + " radians");
					}



				ceil:

					Returns the smallest integer greater than or equal to a value.

					double ceil(double x);

				Arguments:

					* double x: The value whose ceiling is to be calculated.

				Returns:

					double: the smallest integer greater than or equal to x.

				Example:

					```NVGT
					void main() {
						alert("Example", "The ceiling of 3.14 is " + ceil(3.14));
					}



				cos:

					Returns the cosine of an angle given in radians.

					double cos(double x);

				Arguments:

					* double x: The angle (in radians) to get the cosine of.

				Returns:

					double: the cosine of the angle.

				Example:

					```NVGT
					void main() {
						alert("Example", "The cosine of 45 is " + cos(45 * 3.14159 / 180) + " radians");
					}



				cosh:

					Returns the hyperbolic cosine of a value.

					double cosh(double x);

				Arguments:

					* double x: The value whose hyperbolic cosine is to be calculated.

				Returns:

					double: the hyperbolic cosine of the given value.

				Example:

					```NVGT
					void main() {
					    alert("Example", "The hyperbolic cosine of 2 is " + cosh(2));
					}



				floor:

					Returns the largest integer less than or equal to a value.

					double floor(double x);

				Arguments:

					* double x: The value whose floor is to be calculated.

				Returns:

					double: the largest integer less than or equal to x.

				Example:

					```NVGT
					void main() {
						alert("Example", "The floor of 3.14 is " + floor(3.14));
					}



				fraction:

					Returns the fractional part of a value.

					double fraction(double x);

				Arguments:

					* double x: The value whose fractional part is to be extracted.

				Returns:

					double: the fractional part of the given value.

				Example:

					```NVGT
					void main() {
						alert("Example", "The fractional part of 3.75 is " + fraction(3.75));
					}



				log:

					Returns the natural logarithm (base e) of a value.

					double log(double x);

				Arguments:

					* double x: The value whose natural logarithm is to be calculated.

				Returns:

					double: the natural logarithm of the given value.

				Example:

					```NVGT
					void main() {
					    alert("Example", "The natural logarithm of 10 is " + log(10));
					}



				log10:

					Returns the base 10 logarithm of a value.

					double log10(double x);

				Arguments:

					* double x: The value whose base 10 logarithm is to be calculated.

				Returns:

					double: the base 10 logarithm of the given value.

				Example:

					```NVGT
					void main() {
					    alert("Example", "The base 10 logarithm of 100 is " + log10(100));
					}



				pow:

					Returns x raised to the power of y.

					double pow(double x, double y);

				Arguments:

					* double x: The base.

					* double y: The exponent.

				Returns:

					double: x raised to the power of y.

				Example:

					```NVGT
					void main() {
					    alert("Example", "2 raised to the power of 3 is " + pow(2, 3));
					}



				round:

					Returns the value of a number rounded to the nearest integer.

					double round(double n, int p);

				Arguments:

					* double n: The number to be rounded.

					* int p: The number of decimal places to round to.

				Returns:

					double: The value of the number rounded to the nearest integer.

				Example:

					```NVGT
					void main() {
						alert("Example", "Rounding 3.14159 to 2 decimal places gives " + round(3.14159, 2));
					}



				sin:

					Returns the sine of an angle given in radians.

					double sin(double x);

				Arguments:

					* double x: The angle (in radians) to get the sine of.

				Returns:

					double: the sine of the angle.

				Example:

					```NVGT
					void main() {
						alert("Example", "The sine of 45 is " + sin(45 * 3.14159 / 180) + " radians");
					}



				sinh:

					Returns the hyperbolic sine of a value.

					double sinh(double x);

				Arguments:

					* double x: The value whose hyperbolic sine is to be calculated.

				Returns:

					double: the hyperbolic sine of the given value.

				Example:

					```NVGT
					void main() {
					    alert("Example", "The hyperbolic sine of 2 is " + sinh(2));
					}



				sqrt:

					Returns the square root of a value.

					double sqrt(double x);

				Arguments:

					* double x: The value whose square root is to be calculated.

				Returns:

					double: the square root of the given value.

				Example:

					```NVGT
					void main() {
						alert("Example", "The square root of 16 is " + sqrt(16));
					}



				tan:

					Returns the tangent of an angle given in radians.

					double tan(double x);

				Arguments:

					* double x: The angle (in radians) to get the tangent of.

				Returns:

					double: the tangent of the angle.

				Example:

					```NVGT
					void main() {
						alert("Example", "The tangent of 45 is " + tan(45 * 3.14159 / 180) + " radians");
					}



				tanh:

					Returns the hyperbolic tangent of a value.

					double tanh(double x);

				Arguments:

					* double x: The value whose hyperbolic tangent is to be calculated.

				Returns:

					double: the hyperbolic tangent of the given value.

				Example:

					```NVGT
					void main() {
					    alert("Example", "The hyperbolic tangent of 2 is " + tanh(2));
					}



				tinyexpr:

					Evaluate a mathematical expression using the tinyexpr library.

					double tinyexpr(const string&in expression);

				Arguments:

					* const string&in expression: the expression to evaluate.

				Returns:

					double: the result of the expression.

				Example:

					```NVGT
					void main() {
						string expression = input_box("Expression", "Enter expression to evaluate");
						if (expression.is_empty()) exit();
						alert("Result", expression + "= " + tinyexpr(expression));
					}



		Networking:
			classes:
				http:

					Facilitate HTTP requests from a simple, asynchronous interface.

					http();

				Remarks:

					This is one of the easiest ways to perform an http request in NVGT without blocking the rest of your code.

					Simply create an http object, call either it's get, head or post methods, then begin accessing properties on the object such  as progress, status_code or response_body to query and access details of the request.

				Example:

					```NVGT
				include "size.nvgt":
				include "speech.nvgt":
					void main() {
						// This script downloads the latest NVGT windows installer and saves it in the current working directory.
						http h;
						h.get("https://nvgt.zip/windows");
						file f;
						string filename = "";
						show_window("downloading...");
						while (!h.complete) {
							wait(5);
							if (key_pressed(KEY_ESCAPE)) {
								f.close();
								file_delete(filename);
								return; // Destruction of the http object will cancel any requests in progress.
							}
							if (!f.active and h.status_code == 200) { // The file is available.
								string[]@ path = h.url.get_path_segments();
								filename = path.length() > 0? path[-1] : "http.txt";
								f.open(filename, "wb");
							}
							if (f.active) f.write(h.response_body);
							if (key_pressed(KEY_SPACE)) speak(round(h.progress * 100, 2) + "%");
						}
						int keep = question(filename, size_to_string(f.size) + " downloaded, keep file?");
						f.close();
						if (keep != 1) file_delete(filename);
					}



					Methods:
						get:

							Initiate a get request.

							bool get(spec::uri url, name_value_collection@ headers = null, http_credentials@ creds = null);

						Arguments:

							* spec::uri url: A valid URI which must have a scheme of either http or https.

							* name_value_collection@ headers = null: Pairs of extra http headers to pass on to the server.

							* http_credentials@ creds = null: Optional credentials to authenticate the request with.

						Returns:

							bool: True if the request was initiated, or false if there was an error.

						Remarks:

							Once this method is called and returns true, you will usually want to wait until either http.complete returns true or http.running returns false before thoroughly handling the request. However http.progress, http.status_code, http.response_headers and http.response_body are accessible throughout the request's lifecycle to stream the data or get more detailed information.

						Example:

							```NVGT
							void main() {
								// Fetch the latest version of NVGT.
								http h;
								if (!h.get("http://nvgt.dev/downloads/latest_version")) {
									alert("oops", "request failed"); // This class will contain more detailed error management facilities in the near future.
									return;
								}
								h.wait(); // You can also use the h.running or h.complete properties in a loop to execute code in the background while the request progresses.
								alert("nvgt latest version", h.response_body);
							}



						reset:

							Resets the http object to the state it was in at the time of it's construction, canceling any requests in progress.

							void reset();

						Remarks:

							This method waits for the current request to cancel successfully, which is usually instant. However from time to time you could see a small delay as the request thread runs to the next point where it can check the cancelation event.

							This method is implicitly called whenever the http object destructs.

						Example:

							```NVGT
							void main() {
								http h;
								h.get("https://nvgt.dev");
								wait(20);
								h.reset();
								h.get("https://samtupy.com/ip.php");
								h.wait();
								if (h.status_code != 200) alert("oops " + h.status_code, h.response_body);
								else alert("IP address displayed after http reset", h.response_body);
							}



						wait:
							Waits for the current request to complete.

							`void wait();`

						Remarks:
							If no request is in progress, this will return emmedietly. Otherwise, it will block code execution on the calling thread until the request has finished.


					Properties:
						progress:
							Determine the percentage of the request download (from 0.0 to 1.0.

							`const float progress;`

						Remarks:
							This property will be 0 if a request is not in progress or if it is not yet in a state where the content length of the download can be known, and -1 if the progress cannot be determined such as if the remote endpoint does not provide a Content-Length header.


						response_body:
							Read and consume any new data from the current request.

							`const string response_body;`

						Remarks:
							It is important to note that accessing this property will flush the buffer stored in the request. This means that for example you do not want to access http.response_body.length(); because the next time http.response_body is accessed it will be empty or might contain new data.

							Proper usage is to continuously append the value of this property to a string or datastream in a loop while h.running is true or h.complete is false.


				network:
					methods:
						connect:
							Attempt to establish a connection with a server, only works when set up as a client.

							`uint64 network::connect(const string&in hostname, uint16 port);`

						Arguments:
							* const string&in hostname: the hostname/IP address to connect to.
							* uint16 port: the port to use.

						Returns:
							uint64: the peer ID of the connection, or 0 on error.


						destroy:
							Destroys the network object, freeing all its resources, active connections, etc.

							`void network::destroy();`


						disconnect_peer:
							Tell a peer to disconnect and completely disregard its message queue. This means that the peer will be told to disconnect without sending any of its queued packets (if any).

							`bool network::disconnect_peer(uint peer_id);`

						Arguments:
							* uint peer_id: the ID of the peer to disconnect.

						Returns:
							bool: true on success, false otherwise.


						disconnect_peer_forcefully:
							Forcefully disconnect a peer. Unlike `network::disconnect_peer()`, this function doesn't send any sort of notification of the disconnection to the remote peer, instead it closes the connection immediately.

							`bool network::disconnect_peer_forcefully(uint peer_id);`

						Arguments:
							* uint peer_id: the ID of the peer to disconnect.

						Returns:
							bool: true on success, false otherwise.


						disconnect_peer_softly:
							Send a disconnect packet for a peer after sending any remaining packets in the queue and notifying the peer.

							`bool network::disconnect_peer_softly(uint peer_id);`

						Arguments:
							* uint peer_id: the ID of the peer to disconnect.

						Returns:
							bool: true on success, false otherwise.


						get_peer_address:
							Returns the IP address of a particular peer.

							`string network::get_peer_address(uint peer_id);`

						Arguments:
							* uint peer_id: the ID of the peer to check.

						Returns:
							string: the IP address of the specified peer.


						get_peer_list:
							Return a list of all peer ID's currently connected to the server.

							`uint[]@ network::get_peer_list();`

						Returns:
							uint[]@: a handle to an array containing the ID of every peer currently connected to the server.


						request:
							This is the function you'll probably be calling the most when you're dealing with the network object in NVGT. It checks if an event has occurred since the last time it checked. If it has, it returns you a network_event handle with info about it.

							`network_event@ network::request(uint timeout = 0);`

						Arguments:
							* uint timeout = 0: an optional timeout on your packet receiving (see remarks for more information).

						Returns:
							network_event@: a handle to a `network_event` object containing info about the last received event.

						Remarks:
							The timeout parameter is in milliseconds, and it determines how long Enet will wait for new packets before returning control back to the calling application. However, if it receives a packet within the timeout period, it will return and you'll be handed the packet straight away.


						send:
							Attempt to send a packet over the network.

							`bool network::send(uint peer_id, const string&in message, uint8 channel, bool reliable = true);`

						Arguments:
							* uint peer_id: the ID of the peer to send to (specify 1 to send to the server from a client).
							* const string&in message: the message to send.
							* uint8 channel: the channel to send the message on (see the main networking documentation for more details).
							* bool reliable = true: whether or not the packet should be sent reliably or not (see the main networking documentation for more details).

						Returns:
							bool: true if the packet was successfully sent, false otherwise.


						send_reliable:
							Attempt to send a packet over the network reliably.

							`bool network::send_reliable(uint peer_id, const string&in message, uint8 channel);`

						Arguments:
							* uint peer_id: the ID of the peer to send to (specify 1 to send to the server from a client).
							* const string&in message: the message to send.
							* uint8 channel: the channel to send the message on (see the main networking documentation for more details).

						Returns:
							bool: true if the packet was successfully sent, false otherwise.


						send_unreliable:
							Attempt to send a packet over the network unreliably.

							`bool network::send_unreliable(uint peer_id, const string&in message, uint8 channel);`

						Arguments:
							* uint peer_id: the ID of the peer to send to (specify 1 to send to the server from a client).
							* const string&in message: the message to send.
							* uint8 channel: the channel to send the message on (see the main networking documentation for more details).

						Returns:
							bool: true if the packet was successfully sent, false otherwise.


						set_bandwidth_limits:
							Set the incoming and outgoing bandwidth limits of the server (in bytes-per-second).

							`void network::set_bandwidth_limits(uint incoming, uint outgoing);`

						Arguments:
							* uint incoming: the maximum number of allowed incoming bytes per second.
							* uint outgoing: the maximum number of allowed outgoing bytes per second.


						setup_client:
							Sets up the network object as a client.

							`bool network::setup_client(uint8 max_channels, uint16 max_peers);`

						Arguments:
							* uint8 max_channels: the maximum number of channels used on the connection (up to 255).
							* uint16 max_peers: the maximum number of peers allowed by the connection (maximum is 65535).

						Returns:
							bool: true if the client was successfully set up, false otherwise.


						setup_local_server:
							Sets up the network object as a local server on localhost.

							`bool network::setup_local_server(uint16 bind_port, uint8 max_channels, uint16 max_peers);`

						Arguments:
							* uint16 bind_port: the port to bind the server to.
							* uint8 max_channels: the maximum number of channels used on the connection (up to 255).
							* uint16 max_peers: the maximum number of peers allowed by the connection (maximum is 65535).

						Returns:
							bool: true if the server was successfully set up, false otherwise.


						setup_server:
							Sets up the network object as a server.

							`bool network::setup_server(uint16 bind_port, uint8 max_channels, uint16 max_peers);`

						Arguments:
							* uint16 bind_port: the port to bind the server to.
							* uint8 max_channels: the maximum number of channels used on the connection (up to 255).
							* uint16 max_peers: the maximum number of peers allowed by the connection (maximum is 65535).

						Returns:
							bool: true if the server was successfully set up, false otherwise.


					properties:
						active:
							Determine if the network object is active (e.g. a setup_* method has successfully been called on it).

							`bool network::active;`


						bytes_received:
							The number of bytes this network object has received since being set up.

							`uint network::bytes_received;`


						bytes_sent:
							The number of bytes this network object has sent since being set up.

							`uint network::bytes_sent;`


						connected_peers:
							The number of peers currently connected to the server.

							`uint network::connected_peers;`


				network_event:
					This class represents an event received with the network object's `request()` method. It contains information such as the the ID of the peer that sent the event, the message the event was sent on, and the actual packet data itself.


					methods:
						opAssign:
							This class implements the `opAssign()` operator overload, meaning it can be assigned with the "=" operator to another network_event.

							`network_event@ opAssign(network_event@ e);`


					properties:
						channel:
							The channel this event was sent on. See the main networking documentation for more information.

							`uint network_event::channel;`


						message:
							The data associated with this event (AKA the packet).

							`string network_event::message;`


						peer_id:
							The peer ID of the connection this event came from. See the main networking documentation for more information.

							`uint network_event::peer_id;`


						type:
							The type of the network event (see event types for more information).

							`int network_event::type;`


			Constants:
				Event Types:

					This is a list of all the constants supported as event types by NVGT's networking layer.

					- event_none: no event.
					- event_connect: a new user wants to connect.
					- event_disconnect: a user wants to disconnect.
					- event_receive: a user sent a packet.


		Profiling and Debugging:
			Functions:
				assert:

					Evaluates an expression and terminates the script execution if the result is false.

					void assert(bool expression, string message = "");

				Arguments:

					* expression: The expression to evaluate.

					* message: The message to display if false.

				Remarks:

					This is useful for testing and debugging scenarios, allowing you to report a problem.

					In the event that the evaluated expression returns false, an exception is thrown, using the message parameter in the event that the exception is unhandled.

					Please note that, unlike BGT, asserts can actually trigger in release builds.

				Example:

					```NVGT
					void main() {
						assert(false, "I am an assertion violation. I love causing chaos!");
					}



				c_debug_message:

					Print a message to the C debugger.

					void c_debug_message(const string&in message);

				Arguments:

					* const string&in message: The message to print to the C debugger's console.

				Example:

					```NVGT
					void main() {
						c_debug_message("I am a debug message");
						c_debug_break();
					}



				garbage_collect:

					Manually triggers the garbage collector.

					void garbage_collect(bool full = true);

				Arguments:

					full: True for a full cycle, false for a small increment.

				Remarks:

					If full is set to false, the garbage collector only runs a small increment. If you have set the garbage collect mode to manual, you will need to regularly call this function yourself throughout the game. Otherwise this will be done automatically every 5-25 milliseconds, either in a dedicated thread, or during a wait() operation.

					If full is set to true, this will do a full cleanout. This could take considerably longer, especially in larger games, so the automatic system never does this. It is up to you to perform a full cycle at a time that is appropriate for your game.

				Example:

					```NVGT
					void main() {
						garbage_collect();
					}



				generate_profile:

					Returns any data stored in the profiler.

					string generate_profile(bool reset = true);

				Arguments:

					* reset: If true, resets profile data.

				Example:

					```NVGT
					void main() {
						start_profiling();
						wait(1000);
						alert("Profile", generate_profile());
					}



				get_call_stack:

					Get the call stack (list of functions called) at the current point in time in your script.

					string get_call_stack();

				Returns:

					string: a formatted list of the call stack.

				Remarks:

					In the context of a catch block, it's better to use the `last_exception_call_stack` property. If you use `get_call_stack()`, you'll get the callstack where you are in the try statement, as opposed to the call stack that actually caused the exception being caught.

				Example:

					```NVGT
					// Define some example functions.
					void test1() {test2();}
					void test2() {test3();}
					void test3() {
						alert("Call stack", get_call_stack());
					}
					void main() {
						test1();
					}



				get_call_stack_size:

					Get the size of the call stack.

					int get_call_stack_size();

				Returns:

					int: the size of the call stack (i.e. how many functions deep are you currently?)

				Remarks:

					This function doesn't work in the context of exceptions. If you call it in a catch block, you'll most likely get the call stack in the try block, not what actually threw the exception.

				Example:

					```NVGT
					void test1() {test2();}
					void test2() {test3();}
					void test3() {test4();}
					void test4() {
						alert("Call stack size is", get_call_stack_size());
					}
					void main() {
						test1();
					}



				get_exception_file:

					Get the name/path of the source file where the exception occurred.

					string get_exception_file();

				Returns:

					string: the name/path of the file where the exception was thrown.

				Example:

					```NVGT
					void main() {
						try {
							// Throw an index-out-of-bounds error.
							string[] arr = {"a", "test"};
							string item = arr[2];
						}
						catch {
							alert("Exception file path", get_exception_file());
						}
					}



				get_exception_function:

					Get the Angelscript function signature of the function that threw the exception.

					string get_exception_function();

				Returns:

					string: the Angelscript function signature of the throwing function.

				Example:

					```NVGT
					void main() {
						try {
							// Throw an index-out-of-bounds error.
							string[] arr = {"a", "test"};
							string item = arr[2];
						}
						catch {
							alert("Exception function signature", get_exception_function());
						}
					}



				get_exception_info:

					Get informative information about an exception, for example "index-out-of-bounds".

					string get_exception_info();

				Returns:

					string: information about the exception currently being caught.

				Example:

					```NVGT
					void main() {
						try {
							// Throw an index-out-of-bounds error.
							string[] arr = {"a", "test"};
							string item = arr[2];
						}
						catch {
							alert("Exception info", get_exception_info());
						}
					}



				get_exception_line:

					Get the line an exception occurred on.

					int get_exception_line();

				Returns:

					int: the line number the exception occurred on.

				Example:

					```NVGT
					void main() {
						try {
							// Throw an index-out-of-bounds error.
							string[] arr = {"a", "test"};
							string item = arr[2];
						}
						catch {
							alert("Exception line", get_exception_line());
						}
					}



				get_last_error:

					Get the last error flagged by the engine.

					int get_last_error();

				Returns:

					int: the error code that was flagged.

				Remarks:

					This will usually be flagged for recoverable errors in functions that can't return error information directly, such as input and conversion functions, or when more information is available, such as in file or threading operations. More serious errors will usually throw exceptions.

					Please note: At this time there are very few functions that make use of this flag.

				Example:

					```NVGT
					void main() {
						input_box("Test", "Cancel to cause an error"); // Error is set if you cancel.
						alert("Error", get_last_error());
					}



				is_debugger_present:

					Determines if your app is being ran through a debugger.

					bool is_debugger_present();

				Returns:

					bool: true if a debugger is present, false otherwise.

				Example:

					```NVGT
					void main() {
						bool present = is_debugger_present();
						if (present)
							alert("Info", "A debugger is present.");
						else
							alert("Info", "A debugger is not present.");
					}



				reset_profiler:

					Resets the profiler.

					void reset_profiler();

				Example:

					```NVGT
					void main() {
						start_profiling();
						wait(1000);
						reset_profiler(); // Profiler is still running.
						wait(1000);
						alert("Profile", generate_profile());
					}



				start_profiling:

					Starts the profiler.

					void start_profiling();

				Example:

					```NVGT
					void main() {
						start_profiling();
						wait(1000);
						alert("Profile", generate_profile());
					}



				stop_profiling:

					Stops and resets the profiler.

					void stop_profiling();

				Example:

					```NVGT
					void main() {
						start_profiling();
						wait(1000);
						stop_profiling();
						wait(1000);
						start_profiling();
						wait(1000);
						alert("Profile", generate_profile());
					}



				throw:

					Throws an exception with a particular message.

					void throw(const string&in msg);

				Arguments:

					* const string&in msg: the message to include in your exception.

				Remarks:

					This is presently slightly limited, the only value of the exception you can set is the message. There are plans to expand this in the future.

				Example:

					```NVGT
					void main() {
						throw("Something went horribly wrong");
					}



			Global Properties:
				garbage_collect_auto_frequency:

					Controls how often extra garbage collection is performed in automatic modes. Default is 300000 milliseconds (5 minutes). Valid values are from 2000 (2 seconds) to 86400000 (24 hours).

					int garbage_collect_auto_frequency;

				Remarks:

					When this interval has passed, the system will lightly perform extra garbage cleanup.

				Example:

					```NVGT
					void main() {
						alert("GC frequency", garbage_collect_auto_frequency);
					}



				garbage_collect_mode:

					Set or retrieve the mode for the garbage collector. Default is 2 (see remarks below).

					int garbage_collect_mode;

				Remarks:

					This has a range of 1 to 3, where:

					1 is manual (you are responsible for calling the garbage_collect function when convenient).

					2 is automatic (called every wait() cycle on the main thread before sleeping).

					3 is asynchronous (same as 2, but called on a dedicated thread).

				Example:

					```NVGT
					void main() {
						alert("GC mode", garbage_collect_mode);
					}



				last_exception_call_stack:

					Returns the call stack of the currently thrown exception.

					string last_exception_call_stack;

				Remarks:

					In the context of a catch block, it's better to use this property over `get_call_stack()`. In a catch block, `get_call_stack()` will return the callstack where you are in the try statement, as opposed to the call stack that actually caused the exception being caught.

				Example:

					```NVGT
					void test1() {test2();}
					void test2() {throw("I am an exception");}
					void main() {
						try {
							test1();
						}
						catch {
							alert("Call stack", last_exception_call_stack);
						}
					}



				SCRIPT_COMPILED:

					Determine if the script is compiled or not. I.e., are you running from source?

					bool SCRIPT_COMPILED;

				Example:

					```NVGT
					void main() {
						alert("Example", SCRIPT_COMPILED ? "The script is currently compiled" : "The script is not currently compiled");
					}



				SCRIPT_CURRENT_FILE:

					Returns the current file your script is running form.

					string SCRIPT_CURRENT_FILE;

				Example:

					```NVGT
					void main() {
						alert("Your script is contained in", SCRIPT_CURRENT_FILE);
					}



				SCRIPT_CURRENT_FUNCTION:

					Returns the signature of the current function in your script.

					string SCRIPT_CURRENT_FUNCTION;

				Example:

					```NVGT
					void main() {
						alert("The current function is", SCRIPT_CURRENT_FUNCTION);
						random_func("", 0, 0, 0);
					}
					// Function with random unused parameters to showcase the function signatures.
					bool random_func(const string& in a, int b, uint64 c, uint8 = 1) {
						alert("The current function is", SCRIPT_CURRENT_FUNCTION);
						return true;
					}



				SCRIPT_CURRENT_LINE:

					Returns the current line in your script as it's executing.

					string SCRIPT_CURRENT_LINE;

				Example:

					```NVGT
					void main() {
						alert("The current line is", SCRIPT_CURRENT_LINE);
					}



				SCRIPT_EXECUTABLE:

					Returns the path of the executable running your script.

					string SCRIPT_EXECUTABLE;

				Remarks:

					If you're running from source, this will return the path to your NVGT binary.

				Example:

					```NVGT
					void main() {
						alert("The executable path of the script is", SCRIPT_EXECUTABLE);
					}



		Pseudorandom Generation:
			Classes:
				random_interface:
					Defines the class structure that is available in NVGT's object based pseudorandom number generators. A class specifically called random_interface does not exist in the engine, but instead this reference describes methods available in multiple classes that do exist (see remarks).
					1. random_interface();
					2. random_interface(uint seed);
				Arguments (2):
					* uint seed: The number used as the seed/starting point for the RNG, passing the same seed will yield the same sequence of random numbers.
				Remarks:
					NVGT contains several different pseudorandom number generators which can all be instantiated as many times as the programmer needs.

					These generators all share pretty much exactly the same methods by signature and are interacted with in the same way, and so it will be documented only once here. Small topics explaining the differences for each actual generator are documented below this interface.

					These classes all wrap a public domain single header library called [rnd.h](https://github.com/mattiasgustavsson/libs/blob/main/rnd.h) by mattiasgustavsson on Github. The explanations for each generator as well as the following more general explanation about all of them were copied verbatim from the comments in that header, as they are able to describe the generators best and already contain links to more details.

					The library includes four different generators: PCG, WELL, GameRand and XorShift. They all have different characteristics, and you might want to use them for different things. GameRand is very fast, but does not give a great distribution or period length. XorShift is the only one returning a 64-bit value. WELL is an improvement of the often used Mersenne Twister, and has quite a large internal state. PCG is small, fast and has a small state. If you don't have any specific reason, you may default to using PCG.


					Methods:
						next:

							Return the next random number from the generator.

							uint next();

						Returns:

							uint or uint64 depending on generator: The next random number (can be any supported by the integer type used).

						Example:

							```NVGT
							void main() {
								random_pcg r;
								alert("info", r.next());
							}



						nextf:

							Return the next random floating point number from the generator.

							float nextf();

						Returns:

							float: The next random float (between 0.0 and 1.0).

						Example:

							```NVGT
							void main() {
								random_gamerand r;
								alert("info", r.nextf());
							}



						range:

							Return the next random number from the generator with in a minimum and maximum range.

							int range(int min, int max);

						Arguments:

							* int min: The minimum number that could be generated (inclusive).

							* int max: The maximum number that could be generated (inclusive).

						Returns:

							int: A random number within the given range.

						Remarks:

							This function always works using 32 bit integers regardless of the generator used.

						Example:

							```NVGT
							void main() {
								random_xorshift r;
								alert("info", r.range(1, 10));
							}



						seed:

							Seed the random number generator with a new starting point value.

							void seed(uint new_seed = random_seed());

							arguments:

							* uint new_seed = random_seed(): The seed to use (may be uint64 depending on generator, if so will default to random_seed64).

						Remarks:

							All pseudorandom number generators typically need to start from one tiny bit of real-world randomness or obscure value to properly get going.

							By default, nvgt's random number generators are seeded by reading 4 random bytes from the operating system using it's provided API to do so. If you call this seed() function with no arguments, the rng will be re-seeded with a new random seed.

							However, particularly when dealing with game recordings or online play, it can be useful to take control of the random number generator and cause it to replay values. This can be done by setting the seed of an RNG to a predetermined and reused value. This is because the rng is guaranteed to return the same set of numbers for a given seed, as the seed serves as the starting point for the RNG.

							A common method for example may be to fetch the value of the ticks() or random_seed() function yourself, store the result in a variable, and use that variable to seed an rng. If you want someone else on the network to generate the same numbers for your online game, you need only to transmit that same stored ticks() or random_seed() value over the wire and have the receiver also seed the rng to that value. Now both clients are using the same seed, and from that point they will generate random numbers that are the same, so long as of course one end of the party doesn't somehow jump the gun and generate a random number that the other end does not, as now the clients would be out of sync. This is exactly why these random_xxx classes were provided, because it may be needed to generate different random numbers on each client while insuring that some special numbers (such as enemy spawns) always remain the same. Just create multiple random_xxx instances and share the seed with one while doing whatever you want with the other.

						Example:

							```NVGT
								void main() {
									uint seed = random_seed();
									random_well r(seed);
									int num = r.range(1, 100);
									alert("example", "the number is " + num);
									r.seed(seed); // Restore our original seed from before generating the number.
									alert("The new number will still be " + num, r.range(1, 100));
									r.seed(); // internally choose a random seed.
									alert("unknown number", r.range(1, 100));
							}



				random_gamerand:
					GameRand

					Based on the random number generator by Ian C. Bullard:

					http://www.redditmirror.cc/cache/websites/mjolnirstudios.com_7yjlc/mjolnirstudios.com/IanBullard/files/79ffbca75a75720f066d491e9ea935a0-10.html

					GameRand is a random number generator based off an "Image of the Day" posted by Stephan Schaem. More information here:

					http://www.flipcode.com/archives/07-15-2002.shtml


					Look at nvgt's random_interface documentation above to learn how to use this class.

				random_pcg:
					PCG - Permuted Congruential Generator

					PCG is a family of simple fast space-efficient statistically good algorithms for random number generation. Unlike many  general-purpose RNGs, they are also hard to predict.

					More information can be found here: 

					http://www.pcg-random.org/

					Look at nvgt's random_interface documentation above to learn how to use this class.

				random_well:
					WELL - Well Equidistributed Long-period Linear

					Random number generation, using the WELL algorithm by F. Panneton, P. L'Ecuyer and M. Matsumoto.

					More information in the original paper: 

					http://www.iro.umontreal.ca/~panneton/WELLRNG.html

					This code is originally based on WELL512 C/C++ code written by Chris Lomont (published in Game Programming Gems 7) and placed in the public domain. 

					http://lomont.org/Math/Papers/2008/Lomont_PRNG_2008.pdf


					Look at nvgt's random_interface documentation above to learn how to use this class.

				random_xorshift:
					XorShift 

					A random number generator of the type LFSR (linear feedback shift registers). This specific implementation uses the XorShift+ variation, and returns 64-bit random numbers.

					More information can be found here: 

					https://en.wikipedia.org/wiki/Xorshift


					Look at nvgt's random_interface documentation above to learn how to use this class.

			Functions:
				random:

					Generates a pseudorandom number given a range.

					int random(int min, int max);

				Arguments:

					* int min: the minimum possible number to generate.

					* int max: the maximum possible number to generate.

				Returns:

					int: a random number in the given range.

				Remarks:

					Note that this function is a pseudorandom number generator. To learn more, click [here](https://en.wikipedia.org/wiki/Pseudorandom_number_generator).

				Example:

					```NVGT
					void main() {
						alert("Example", random(1, 100));
					}



				random_bool:

					Generates a true or false value pseudorandomly.

					bool random_bool(int percent = 50);

				Arguments:

					* int percent = 50: the likelihood (as a percent from 0-100) of generating a true value. Defaults to 50.

				Returns:

					bool: either true or false.

				Remarks;:

					Note that this function is a pseudorandom number generator. To learn more, click [here](https://en.wikipedia.org/wiki/Pseudorandom_number_generator).

				Example:

					```NVGT
					void main() {
						alert("Example", random_bool() ? "true" : "false");
					}



				random_character:

					Generates a random ascii character.

					string random_character(const string&in min_ascii, const string&in max_ascii);

				Arguments:

					* const string&in min_ascii: the minimum possible ascii character to be generated.

					* const string&in max_ascii: the maximum possible ascii character.

				Returns:

					string: a random ascii character in the given range.

				Remarks:

					If you pass a string with more than one byte in it to either of the arguments in this function, only the first byte is used.

					Note that this function uses a pseudorandom number generator. To learn more, click [here](https://en.wikipedia.org/wiki/Pseudorandom_number_generator).

				Example:

					```NVGT
					void main() {
						alert("Your random alphabetical character is", random_character("a", "z"));
					}



				random_seed:

					Returns 4 or 8 random bytes from the operating system usually used for seeding random number generators.

					1. uint random_seed();

					2. uint64 random_seed64();

				Returns (1):

					uint: A 4 byte random number.

				Returns (2):

					uint64: An 8 byte random number.

				Remarks:

					A more detailed description on seeding random number generators is in the documentation for the random_interface::seed function.

					To retrieve the random bytes in the first place, this function uses cryptographic APIs on windows and /dev/urandom on unix.

				Example:

					```NVGT
					void main() {
						uint seed = random_seed();
						alert("32 bit seed", seed);
						uint64 seed64 = random_seed64();
						alert("64 bit seed", seed64);
					}



		Security:
			Functions:
				string_aes_decrypt:

					Decrypts a string using the AES 256 bit CBC algorithm.

					string string_aes_decrypt(const string&in the_data, const string&in the_key)

				Arguments:

					* const string&in the_data: The data to be decrypted.

					* const string&in the_key: The key to decrypt the data with.

				Returns:

					string: decrypted text on success or an empty string on failure.

				Remarks:

					The Advanced Encryption Standard (AES) is a publicly available algorithm for encrypting data which is trusted world wide at the time of implementation into NVGT. This algorithm takes as input some plain text or binary data, then converts that data into unreadable bytes given an encryption key known only to the programmer. Only with the correct encryption key can the data returned from this function be converted (deciphered) back into the original, unencrypted data.

					For anyone interested in more details or who wants more control over the AES encryption such as by providing your own initialization vector, see the aes_encrypt datastream class which gives this control. In the case of this high level encryption function, NVGT will derive an initialization vector from the encryption key provided using hashes, bitwise math, and any other custom security functions a c++ programmer wishes to add to their own version of NVGT. For most users, this is not a concern and most will not even need to know what an initialization vector is to safely use the encryption provided by this function.

				Example:

					```NVGT
					void main() {
						string encrypted_text = string_aes_encrypt(string_base64_decode("SGVsbG8sIEkgYW0gYSBzdHJpbmch"), "nvgt_example_key_for_docs"); // We encrypt our example string after decoding it with base64; only encoded as such to prevent seeing unreadable characters.
						string decrypted_text = string_aes_decrypt(encrypted_text, "wrong_nvgt_example_key_for_docs");
						// Show that the text could not be decrypted because the wrong key was provided.
						alert("example", decrypted_text); // The dialog will be empty because decrypted_text is an empty string due to decryption failure.
						// Now show how the text can indeed be decrypted with the proper key.
						decrypted_text = string_aes_decrypt(encrypted_text, "nvgt_example_key_for_docs");
						alert("example", decrypted_text);
					}



				string_aes_encrypt:

					Encrypts a string using the AES 256 bit CBC algorithm.

					string string_aes_encrypt(const string&in the_data, const string&in the_key)

				Arguments:

					* const string&in the_data: The data to be encrypted.

					* const string&in the_key: The key to encrypt the data with.

				Returns:

					string: Encrypted ciphertext on success or an empty string on failure.

				Remarks:

					The Advanced Encryption Standard (AES) is a publicly available algorithm for encrypting data which is trusted world wide at the time of implementation into NVGT. This algorithm takes as input some plain text or binary data, then converts that data into unreadable bytes given an encryption key known only to the programmer. Only with the correct encryption key can the data returned from this function be converted (deciphered) back into the original, unencrypted data.

					For anyone interested in more details or who wants more control over the AES encryption such as by providing your own initialization vector, see the aes_encrypt datastream class which gives this control. In the case of this high level encryption function, NVGT will derive an initialization vector from the encryption key provided using hashes, bitwise math, and any other custom security functions a c++ programmer wishes to add to their own version of NVGT. For most users, this is not a concern and most will not even need to know what an initialization vector is to safely use the encryption provided by this function.

				Example:

					```NVGT
					void main() {
						string plaintext = "Hello, I am a string!";
						string encrypted_text = string_aes_encrypt(plaintext, "nvgt_example_key_for_docs");
						// Prove that the text was encrypted by showing a base64 encoded version of it to the user. We encode the encrypted data for display because it may contain unprintable characters after encryption.
						alert("example", string_base64_encode(encrypted_text));
						string decrypted_text = string_aes_decrypt(encrypted_text, "wrong_nvgt_example_key_for_docs");
						// Show that the text could not be decrypted because the wrong key was provided.
						alert("example", decrypted_text); // The dialog will be empty because decrypted_text is an empty string due to decryption failure.
						// Now show how the text can indeed be decrypted with the proper key.
						decrypted_text = string_aes_decrypt(encrypted_text, "nvgt_example_key_for_docs");
						alert("example", decrypted_text);
					}



			Global Properties:
				memory_scan_detected:
					Determines if an application is scanning the address space of your game.

					```nvgt
					const atomic_flag memory_scan_detected;

				Remarks:
					This flag is set internally by the engine whenever `wait()` or `refresh_window()` are called and it can be determined by the engine that an external process is reading the virtual address space of the process. In some circumstances, writing to the address space by an external process may trigger this check. It is strongly recommended that you perform this check immediately after invoking either `wait()` or `refresh_window()`, as the flag is volatile and may change from iteration to iteration.


				speed_hack_detected:
					Attempts to determine if speed-hacking technology is being used.

					```nvgt
					const atomic_flag speed_hack_detected;

				Remarks:
					This check is performed by the engine every time `wait()` and `refresh_window()` is called and this flag is set whenever the engine can determine that speed-hacking technology is being used. Speed-hacking typically works by hooking low-level operating system timer and time measurement routines to alter the duration of a tick. For the purposes of this description, a tick is an operating system-defined duration which determines how quickly a monotonically increasing timer routine's internal value changes every time that value is requested by the engine. It is strongly recommended that you perform this check immediately after invoking either `wait()` or `refresh_window()`, as the flag is volatile and may change from iteration to iteration.

				Warning:
					This check is an extention to, and NOT a substitute for, existing speed-hacking detection technology employed by the engine and by any technology you incorporate into your own game. It MUST NOT be solely relied upon to determine if speed-hacking technology is being used. Although the engine does attempt to be as comprehensive as possible without impacting performance to an unreasonable degree, any reasonably advanced adversary will be able to bypass this check.

		Text-To-Speech:
			Text to speech:
				This section contains references for the functionality that allows for speech output. Both direct speech engine support is available as well as outputting to screen readers.

			Notes on screen Reader Speech functions:
				This set of functions lets you output to virtually any screen reader, either through speech, braille, or both. In addition, you can query the availability of speech/braille in your given screen reader, get the name of the active screen reader, and much more!

				On Windows, we use the [Tolk](https://github.com/dkager/tolk) library. If you want screen reader speech to work on Windows, you have to make sure Tolk.dll, nvdaControllerClient64.dll, and SAAPI64.dll are in the lib folder you ship with your games.


			classes:
				tts_voice:

					This class provides a convenient way to communicate with all of the text-to-speech voices a user has installed on their system.

					tts_voice();

				Remarks:

					NVGT implements a tiny [builtin fallback speech synthesizer](https://github.com/mattiasgustavsson/libs/blob/main/speech.h) that is used either when there is no speech engine available on the system, or if it is explicitly set with the tts_voice::set_voice() function. It sounds terrible, and is only intended for emergencies, after all understandable speech is better than no speech at all. It was used heavily when porting NVGT to other platforms, because it allowed focus to be primarily fixed on getting nvgt to run while individual tts engine support on each platform could be added later. If you don't want your end users to be able to select this speech synthesizer, pass a blank string to this constructor.

				Example:

					```NVGT
					void main() {
						tts_voice v;
						v.speak_wait("Type some text and I'll repeat it. Leave the field blank to stop");
						string text = "";
						while (true) {
							text = input_box("Text", "Type some text to have the TTS speak.");
							if (text.empty()) break;
							v.speak(text);
						}
					}



					Methods:
						get_speaking:

							Check if the text to speech voice is currently speaking.

							bool tts_voice::get_speaking();

						Returns:

							* bool: whether the tts voice is speaking. True if so, false if not.

						Example:

							```NVGT
							void main() {
								tts_voice v;
								alert("Is it speaking right now?", v.get_speaking());
								v.speak("I'm now saying something, I think...");
								wait(250);
								alert("How about now?", v.get_speaking());
							}



						get_voice_count:

							Get a count of all available text to speech voices, including the builtin.

							int tts_voice::get_voice_count();

						Returns:

							* int: the number of tts voices.

						Remarks:

							The default voice built into Nvgt is counted here so it's always one more than the available system voices.

						Example:

							```NVGT
							void main() {
								tts_voice v;
								alert("result:", "You have "+v.get_voice_count()+" voices");
							}



						get_voice_name:

							Get the name of a voice at a particular index.

							string tts_voice::get_voice_name(int index);

						Parameters:

							* int index: the index of the voice to get the name of.

						Returns:

							string: the name of the voice.

						Example:

							```NVGT
							void main() {
								tts_voice v;
								if (v.voice_count < 1) {
									alert("Oh no", "Your system does not appear to have any TTS voices.");
									exit();
								}
								alert("The first voice on your system is", v.get_voice_name(0));
							}



						get_volume:

							Get the text to speech voice's volume (how loud it is).

							int tts_voice::get_volume();

						Returns:

							int: the current volume in percentage.

						Remarks:

							0 is silent, 100 is loudest. If this hasn't been assigned, it will use the OS setting which may not be 100%. Beware if running old code, this is different from bgt having 0 be the max.

						Example:

							```NVGT
							void main() {
								tts_voice v;
								v.set_volume(50);
								alert("the current volume is: ", ""+v.get_volume());
							}



						list_voices:

							List all the available voices of the tts_voice object.

							string[]@ tts_voice::list_voices();

						Returns:

							string[]@: a handle to an array containing all the voice names (as strings).

						Example:

							```NVGT
							void main() {
								tts_voice v;
								string[]@ voices = v.list_voices();
								alert("Your available text-to-speech voices are", join(voices, ", "));
							}



						refresh:
							Refreshes the list of TTS voices installed on the system this object knows about.

							`bool tts_voice::refresh();`

						Returns:
							bool: true if the list was successfully refreshed, false otherwise.


						set_rate:

							Set the text to speech voice's playback rate (how fast it speaks).

							void tts_voice::set_rate(int rate);

						Arguments:

							* int rate: the desired speech rate.

						Remarks:

							-10 is slowest, 10 is fastest, with 0 being default speed, or 50 percent if you like. If this hasn't been manually assigned, it will use the OS setting which may not be 0. If moving in old bgt code, you may want to update it to use this method instead of the rate property.

						Example:

							```NVGT
							void main() {
								tts_voice v;
								v.set_rate(0);
								v.speak("This is at fifty percent rate");
								v.set_rate(5);
								v.speak_wait("this is at seventy-five percent rate");
								//demonstrate setting it like it's a percentage, but note that it'll jump a bit because percentages have 100 values but there are only technically 21 choices, so implement this idea with an increment of 5 for best results.
								uint desired_rate=85;
								v.set_rate(desired_rate/5-10);
								int resulting_rate=v.get_rate();
								alert("after setting as 85 percent, the resulting rate now is:", resulting_rate);
							}



						set_voice:
							Set the voice of the TTS voice.

							`bool tts_voice::set_voice(int index);`

						Arguments:
							* int index: index of the voice to switch to.

						Returns:
							bool: true if the voice was successfully set, false otherwise.


						set_volume:

							Set the text to speech voice's volume (how loud it is).

							void tts_voice::set_volume(int volume);

						Arguments:

							* int volume: the desired volume level.

						Remarks:

							0 is silent, 100 is loudest. If this hasn't been assigned, it will use the OS setting which may not be 100%. Beware if running old code, this is different from bgt having 0 be the max. It's better to use this instead of directly setting volume property.

						Example:

							```NVGT
							void main() {
								tts_voice v;
								v.set_volume(50);
								v.speak("This is at 50 volume");
								v.set_volume(100);
								v.speak_wait("this is at 100 volume");
							}



						speak:

							Speak a string of text through the currently active TTS voice.

							bool tts_voice::speak(const string &in text, bool interrupt = false);

						Arguments:

							* const string&in text: the text to be spoken.

							* bool interrupt = false: whether or not to interrupt the previously speaking speech to speak the new string.

						Returns:

							bool: true if speech was successful, false otherwise.

						Example:

							```NVGT
							void main() {
								tts_voice v;
								v.speak("Hello, world!");
								while (v.speaking) {}
							}



						speak_interrupt:

							Speaks a string (forcefully interrupting) through the currently active text-to-speech voice.

							bool tts_voice::speak_interrupt(const string&in text);

						Arguments:

							* const string&in text: the text to speak.

						Returns:

							bool: true if the speech was generated successfully, false otherwise.

						Example:

							```NVGT
							void main() {
								int orig_ticks = ticks();
								tts_voice v;
								v.speak("This is an incredibly long string that will eventually get cut off by a much worse one.");
								while (v.speaking) {
									if ((ticks() - orig_ticks) > 250)
										break;
								}
								v.speak_interrupt("You have been interrupted!");
								while (v.speaking) {}
							}



						speak_to_file:

							Outputs a string of text through the currently active TTS voice to a wave file.

							bool tts_voice::speak_to_file(const string&in filename, const string&in text);

						Arguments:

							* const string&in filename: the filename to write to.

							* const string&in text: the text to synthesize.

						Returns:

							bool: true if synthesis was successful and the file was written, false otherwise.

						Example:

							```NVGT
							void main() {
								// This is actually quite broken currently, it only ever writes  a 4 KB wav file. I think the problem is probably in speaking to memory in general, also see test/speak_to_file.nvgt.
								string filename = input_box("Filename", "Enter the name of the file to save to (without the .wav extension).");
								if (filename.is_empty()) {
									alert("Error", "You didn't type a filename.");
									exit();
								}
								string text = input_box("Text", "Enter the text to synthesize.");
								if (text.is_empty()) {
									alert("Error", "You didn't type any text.");
									exit();
								}
								tts_voice v;
								v.speak_to_file(filename + ".wav", text);
								alert("Info", "Done!");
							}



						stop:

							Stop the TTS voice, if currently speaking.

							bool tts_voice::stop();

						Returns:

							bool: true if the voice was successfully stopped, false otherwise.

						Example:

							```NVGT
							void main() {
								tts_voice v;
								v.speak("Hello there!");
								wait(500);
								v.stop();
							}



					Properties:
						speaking:

							Is the tts_voice currently speaking?

							bool tts_voice::speaking;

						Example:

							```NVGT
							void main() {
								tts_voice v;
								v.speak("Hello there! This is a very long string that will hopefully be spoken for at least a couple seconds");
								wait(500);
								alert("Example", v.speaking ? "The TTS voice is currently speaking. Press OK to stop it" : "The TTS voice is not currently speaking");
							}



						voice:

							The number of the currently selected voice.

							int tts_voice::voice;

						Example:

							```NVGT
							void main() {
								tts_voice v;
								alert("Current voice index", v.voice);
								v.set_voice(0);
								alert("Current voice index", v.voice);
							}



						voice_count:

							Represents the total number of available tex-to-speech voices on your system.

							int tts_voice::voice_count;

						Example:

							```NVGT
							void main() {
								tts_voice v;
								alert("Your system has", v.voice_count + " text-to-speech " + (v.voice_count == 1 ? "voice" : "voices"));
							}



			Functions:
				screen_reader_braille:

					Brailles a string through the currently active screen reader, if supported.

					bool screen_reader_braille(const string&in text);

				Arguments:

					* const string&in text: the text to braille.

				Returns:

					bool: true if the function succeeded, false otherwise.

				Example:

					```NVGT
					void main() {
						screen_reader_braille("This message will only be brailled.");
					}



				screen_reader_detect:

					Returns the name of the currently active screen reader as a string.

					string screen_reader_detect();

				Returns:

					string: the name of the given screen reader, or a blank string if none was found.

				Example:

					```NVGT
					void main() {
						string sr = screen_reader_detect();
						if (sr.is_empty())
							alert("Info", "No screen reader found.");
						else
							alert("Info", "The active screen reader is " + sr);
					}



				screen_reader_has_braille:

					Determine if the active screen reader supports braille output.

					bool screen_reader_has_braille();

				Returns:

					bool: true if the active screen reader supports braille, false otherwise.

				Example:

					```NVGT
					void main() {
						bool has_braille  = screen_reader_has_braille();
						if (has_braille)
							alert("Info", "The currently active screen reader supports braille.");
						else
							alert("Info", "The currently active screen reader does not support braille.");
					}



				screen_reader_has_speech:

					Determines if the active screen reader supports speech output.

					bool screen_reader_has_speech();

				Returns:

					bool: true if the screen reader supports speech, false if not, or if no screen reader is active.

				Example:

					```NVGT
					void main() {
						bool has_speech = screen_reader_has_speech();
						if (has_speech)
							alert("Info", "The active screen reader supports speech.");
						else
							alert("Info", "The active screen reader does not support speech.");
					}



				screen_reader_is_speaking:

					Determine if the currently active screen reader (if any) is currently speaking.

					bool screen_reader_is_speaking();

				Returns:

					bool: true if the currently active screen reader is speaking, false otherwise.

				Remarks:

					This function isn't foolproof, as some screen readers (e.g. NVDA) don't have a way to query this information.

				Example:

					```NVGT
					void main() {
						screen_reader_speak("Hello there, this is a very long string that will hopefully be spoken!", true);
						wait(50);
						bool speaking = screen_reader_is_speaking();
						if (speaking)
							alert("Info", "The screen reader is speaking.");
						else
							alert("Info", "The screen reader is not speaking.");
					}



				screen_reader_output:

					Speaks and brailles a string through the currently active screen reader, if supported.

					bool screen_reader_output(const string&in text, bool interrupt = true);

				Arguments:

					* const string&in text: the text to output.

					* bool interrupt = true: Whether or not the previously spoken speech should be interrupted or not when speaking the new string.

				Returns:

					bool: true if the function succeeded, false otherwise.

				Example:

					```NVGT
					void main() {
						screen_reader_output("This message will be both spoken and brailled!", true);
					}



				screen_reader_speak:

					Speaks a string through the currently active screen reader, if supported.

					bool screen_reader_speak(const string&in text, bool interrupt = true);

				Arguments:

					* const string&in text: the text to speak.

					* bool interrupt = true: Whether or not the previously spoken speech should be interrupted or not when speaking the new string.

				Returns:

					bool: true if the function succeeded, false otherwise.

				Example:

					```NVGT
					void main() {
						screen_reader_speak("This message will only be spoken.", true);
					}



			Global Properties:
				SCREEN_READER_AVAILABLE:

					reports if a screen reader is currently available.

					const bool SCREEN_READER_AVAILABLE;

				Example:

					```NVGT
					void main() {
						if (SCREEN_READER_AVAILABLE)
							alert("Info", "A screen reader is available.");
						else
							alert("Info", "A screen reader is not available.");
					}



		User Interface:
			This section contains all functions that involve directly interacting with the user, be that by showing a window, checking the state of a key, or placing data onto the users clipboard. These handle generic input such as from the keyboard, and also generic output such as an alert box.

		Window management notes:
			* You can only have one NVGT window showing at a time.
			* without an nvgt window, there is no way to capture keyboard and mouse input from the user.
			* It is strongly advised to always put `wait(5);` in your main loops. This prevents your game from hogging the CPU, and also prevents weird bugs with the game window sometimes freezing etc.


			Enums:
				key_code:
					This is a complete list of possible keycodes in NVGT, as well as a short description of what they are. These are registered in the key_code enum, meaning you can use this type to pass any value listed here around though it is also safe to use unsigned integers.

				Letter keys:
					* KEY_UNKNOWN: unknown.
					* KEY_A: the A key.
					* KEY_B: the B key.
					* KEY_C: the C key.
					* KEY_D: the D key.
					* KEY_E: the E key.
					* KEY_F: the F key.
					* KEY_G: the G key.
					* KEY_H: the H key.
					* KEY_I: the I key.
					* KEY_J: the J key.
					* KEY_K: the K key.
					* KEY_L: the L key.
					* KEY_M: the M key.
					* KEY_N: the N key.
					* KEY_O: the O key.
					* KEY_P: the P key.
					* KEY_Q: the Q key.
					* KEY_R: the R key.
					* KEY_S: the S key.
					* KEY_T: the T key.
					* KEY_U: the U key.
					* KEY_V: the V key.
					* KEY_W: the W key.
					* KEY_X: the X key.
					* KEY_Y: the Y key.
					* KEY_Z: the Z key.

				Number keys:
					* KEY_1: the 1 key.
					* KEY_2: the 2 key.
					* KEY_3: the 3 key.
					* KEY_4: the 4 key.
					* KEY_5: the 5 key.
					* KEY_6: the 6 key.
					* KEY_7: the 7 key.
					* KEY_8: the 8 key.
					* KEY_9: the 9 key.
					* KEY_0: the 0 key.

				Special keys:
					* KEY_RETURN: the Return (or enter) key.
					* KEY_ESCAPE: the Escape key.
					* KEY_BACK: the Backspace key.
					* KEY_TAB: the Tab key.
					* KEY_SPACE: the Space key.
					* KEY_MINUS: the Minus (dash) key.
					* KEY_EQUALS: the Equals key.
					* KEY_LEFTBRACKET: the Left Bracket key.
					* KEY_RIGHTBRACKET: the Right Bracket key.
					* KEY_BACKSLASH: the Backslash key.
					* KEY_NONUSHASH: the Non-US Hash key.
					* KEY_SEMICOLON: the Semicolon key.
					* KEY_APOSTROPHE: the Apostrophe key.
					* KEY_GRAVE: the Grave key.
					* KEY_COMMA: the Comma key.
					* KEY_PERIOD: the Period key.
					* KEY_SLASH: the Slash key.
					* KEY_CAPSLOCK: the Caps Lock key.

				Function keys:
					* KEY_F1: the F1 key.
					* KEY_F2: the F2 key.
					* KEY_F3: the F3 key.
					* KEY_F4: the F4 key.
					* KEY_F5: the F5 key.
					* KEY_F6: the F6 key.
					* KEY_F7: the F7 key.
					* KEY_F8: the F8 key.
					* KEY_F9: the F9 key.
					* KEY_F10: the F10 key.
					* KEY_F11: the F11 key.
					* KEY_F12: the F12 key.
					* KEY_F13: the F13 key.
					* KEY_F14: the F14 key.
					* KEY_F15: the F15 key.
					* KEY_F16: the F16 key.
					* KEY_F17: the F17 key.
					* KEY_F18: the F18 key.
					* KEY_F19: the F19 key.
					* KEY_F20: the F20 key.
					* KEY_F21: the F21 key.
					* KEY_F22: the F22 key.
					* KEY_F23: the F23 key.
					* KEY_F24: the F24 key.

				Arrow keys:
					* KEY_RIGHT: the Right Arrow key.
					* KEY_LEFT: the Left Arrow key.
					* KEY_DOWN: the Down Arrow key.
					* KEY_UP: the Up Arrow key.

				Numpad keys:
					* KEY_NUMLOCKCLEAR: the Num Lock key.
					* KEY_NUMPAD_DIVIDE: the numpad divide key.
					* KEY_NUMPAD_MULTIPLY: the numpad multiply key.
					* KEY_NUMPAD_MINUS: the numpad minus key.
					* KEY_NUMPAD_PLUS: the numpad plus key.
					* KEY_NUMPAD_ENTER: the numpad enter key.
					* KEY_NUMPAD_1: the Numpad 1 key.
					* KEY_NUMPAD_2: the Numpad 2 key.
					* KEY_NUMPAD_3: the Numpad 3 key.
					* KEY_NUMPAD_4: the Numpad 4 key.
					* KEY_NUMPAD_5: the Numpad 5 key.
					* KEY_NUMPAD_6: the Numpad 6 key.
					* KEY_NUMPAD_7: the Numpad 7 key.
					* KEY_NUMPAD_8: the Numpad 8 key.
					* KEY_NUMPAD_9: the Numpad 9 key.
					* KEY_NUMPAD_0: the Numpad 0 key.
					* KEY_NUMPAD_PERIOD: the Numpad Period key.

				Modifier keys:
					* KEY_LCTRL: the Left Control key.
					* KEY_LSHIFT: the Left Shift key.
					* KEY_LALT: the Left Alt key.
					* KEY_LGUI: the left windows/command/super key (depending on platform).
					* KEY_RCTRL: the Right Control key.
					* KEY_RSHIFT: the Right Shift key.
					* KEY_RALT: the Right Alt key.
					* KEY_RGUI: the right windows/command/super key (depending on platform).

				Miscellaneous keys:
					* KEY_MODE: the Mode key.
					* KEY_APPLICATION: the Application key.
					* KEY_POWER: the Power key.
					* KEY_PRINTSCREEN: the Print Screen key.
					* KEY_SCROLLLOCK: the Scroll Lock key.
					* KEY_PAUSE: the Pause key.
					* KEY_INSERT: the Insert key.
					* KEY_HOME: the Home key.
					* KEY_PAGEUP: the Page Up key.
					* KEY_DELETE: the Delete key.
					* KEY_END: the End key.
					* KEY_PAGEDOWN: the Page Down key.

				Media keys:
					* KEY_MUTE: the Mute key.
					* KEY_VOLUMEUP: the Volume Up key.
					* KEY_VOLUMEDOWN: the Volume Down key.
					* KEY_MEDIA_NEXT_TRACK: the next track key.
					* KEY_MEDIA_PREVIOUS_TRACK: the previous track key.
					* KEY_MEDIA_STOP: the stop media key.
					* KEY_MEDIA_PLAY: the play media key.
					* KEY_MUTE: the mute key.
					* KEY_MEDIA_SELECT: the media select key.

				Browser and Application keys:
					* KEY_AC_SEARCH: the AC Search key.
					* KEY_AC_HOME: the AC Home key.
					* KEY_AC_BACK: the AC Back key.
					* KEY_AC_FORWARD: the AC Forward key.
					* KEY_AC_STOP: the AC Stop key.
					* KEY_AC_REFRESH: the AC Refresh key.
					* KEY_AC_BOOKMARKS: the AC Bookmarks key.

				Additional keys:
					* KEY_MEDIA_EJECT: the eject key.
					* KEY_SLEEP: the sleep key.
					* KEY_MEDIA_REWIND: the media rewind key.
					* KEY_MEDIA_FAST_FORWARD: the media fast forward key.
					* KEY_SOFTLEFT: the Soft Left key.
					* KEY_SOFTRIGHT: the Soft Right key.
					* KEY_CALL: the Call key.
					* KEY_ENDCALL: the End Call key.
					* KEY_AC_SEARCH: the AC Search key.
					* KEY_AC_HOME: the AC Home key.
					* KEY_AC_BACK: the AC Back key.
					* KEY_AC_FORWARD: the AC Forward key.
					* KEY_AC_STOP: the AC Stop key.
					* KEY_AC_REFRESH: the AC Refresh key.
					* KEY_AC_BOOKMARKS: the AC Bookmarks key.


				key_modifier:

					This is a complete list of supported key modifiers and their descriptions. These are registered in the key_modifier enum, so you can use that type to pass any value listed here around.

					* KEYMOD_NONE: no modifier.

					* KEYMOD_LSHIFT: left shift key.

					* KEYMOD_RSHIFT: right shift key.

					* KEYMOD_LCTRL: left control key.

					* KEYMOD_RCTRL: right control key.

					* KEYMOD_LALT: left alt key.

					* KEYMOD_RALT: right alt key.

					* KEYMOD_LGUI: left windows/command/super key (depending on platform).

					* KEYMOD_RGUI: right windows/command/super key (depends on platform).

					* KEYMOD_NUM: numlock key.

					* KEYMOD_CAPS: capslock key.

					* KEYMOD_MODE: input switch mode (only on certain keyboards).

					* KEYMOD_SCROLL: scroll lock key.

					* KEYMOD_CTRL: either control key.

					* KEYMOD_SHIFT: either shift key.

					* KEYMOD_ALT: either alt key.

					* KEYMOD_GUI: either windows/command/super key.

				Example:

					```NVGT
					void main() {
						show_window("Example");
						wait(50); // Give the screen readers enough time to speak the window title before speaking.
						screen_reader_output("Press alt+f4 to close this window.", true);
						while (true) {
							wait(5);
							if (keyboard_modifiers & KEYMOD_ALT > 0 && key_pressed(KEY_F4))
								exit();
						}
					}



				message_box_flags:
					This is an enumeration of possible flags to be passed to the message_box(), alert(), question(), and similar functions.

					* MESSAGE_BOX_ERROR: the message box should act like an error dialog.
					* MESSAGE_BOX_WARNING: the message box should act like a warning dialog.
					* MESSAGE_BOX_INFORMATION: the message box should act like an informative dialog.
					* MESSAGE_BOX_BUTTONS_LEFT_TO_RIGHT: arrange the buttons in the dialog from left to right.
					* MESSAGE_BOX_BUTTONS_RIGHT_TO_LEFT: arrange the buttons from right-to-left in the message box.


				touch_device_type:
					These are the possible types of touch input devices.

					* TOUCH_DEVICE_INVALID = -1: Returned if an invalid ID is passed to get_touch_device_type.
					* TOUCH_DEVICE_DIRECT: A touch screen with window-relative coordinates.
					* TOUCH_DEVICE_INDIRECT_ABSOLUTE: A trackpad with absolute device coordinates.
					* TOUCH_DEVICE_INDIRECT_RELATIVE: A trackpad with screen cursor-relative coordinates.


				window_flags:
					An enumeration of flags that can be used with the show_window function to control the window's appearance.

					* WINDOW_FLAG_FULLSCREEN: The window will be in true fullscreen mode, taking up the entire screen.
					* WINDOW_FLAG_RESIZABLE: The window will be resizable by the user.
					* WINDOW_FLAG_MAXIMIZED: The window will be maximized to fill the screen, but will still have a title bar and borders.
					* WINDOW_FLAG_MINIMIZED: The window will be created in a minimized state.
					* WINDOW_FLAG_HIDDEN: The window will be hidden upon creation.
					* WINDOW_FLAG_BORDERLESS: The window will have no border or title bar.
					* WINDOW_FLAG_ALWAYS_ON_TOP: The window will always be on top of other windows.


			Functions:
				alert:

					Display a message box with an OK button to the user.

					int alert(const string&in title, const string&in text, bool can_cancel = false, uint flags = 0);

				Arguments:

					* const string&in title: the title of the dialog.

					* const string&in text: the text of the dialog.

					* bool can_cancel = false: determines if a cancel button is present.

					* uint flags = 0: a combination of flags (see message_box_flags for more information).

				Returns:

					int: the number of the button that was pressed, either OK or cancel.

				Example:

					```NVGT
					void main() {
						alert("Hello", "I am a standard alert");
						alert("Hi", "And I'm a cancelable one", true);
					}



				android_request_permission:
					Request a permission from the Android operating system either synchronously or asynchronously.

					`bool android_request_permission(string permission, android_permission_request_callback@ callback = null, string callback_data = "");`

				Arguments:
					* string permission: A permission identifier such as android.permission.RECORD_AUDIO.
					* android_permission_request_callback@ callback = null: An optional function that should be called when the user responds to the permission request (see remarks).
					* string callback_data = "": An arbitrary string that is passed to the given callback function.

				Returns:
					bool: true if the request succeeded (see remarks), false otherwise or if this function was not called on Android.

				Remarks:
					The callback signature this function expects is registered as :
					`funcdef void android_permission_request_callback(string permission, bool granted, string user_data);`

					This function behaves very differently depending on whether you've provided a callback or not. If you do not, this function blocks the thread that called it until the user responds to the permission request, in which case it returns true or false depending on whether the user has actually granted the permission. On the other hand if you do provide a callback, the instant return value of this function only indicates whether the request has been made rather than whether the permission has been granted, as that information will be provided later in the given callback.

					Beware that the callback you provide might get invoked on any thread, thus the provision of a more simple, blocking alternative. If you do provide a callback, you are responsible for handling any data races that may result.


				android_show_toast:
					Shows an Android toast notification, small popups that are unique to that operating system.

					`bool android_show_toast(const string&in message, int duration, int gravity = -1, int x_offset = 0, int y_offset = 0);`

				Arguments:
					* const string&in message: The message to display.
					* int duration: 0 for short or 1 for long, all other values are undefined.
					* int gravity = -1: One of [the values on this page](https://developer.android.com/reference/android/view/Gravity) or -1 for no preference.
					* int x_offset = 0, int y_offset = 0: Only used if gravity is set, see Android developer documentation.

				Returns:
					bool: true if the toast was shown, false otherwise or if called on a platform other than Android.

				Remarks:
					[Learn more about android toasts notifications here.](https://developer.android.com/guide/topics/ui/notifiers/toasts)


				clipboard_get_text:

					Returns the text currently on the user's clipboard.

					string clipboard_get_text();

				Returns:

					string: The text on the user's clipboard, as UTF_8.

				Example:

					```NVGT
					void main() {
						string text = clipboard_get_text();
						if (text == "")
							alert("Info", "Your clipboard is empty");
						else
							if (text.length() > 1024)
								alert("Info", "Your clipboard contains a long string. It is " + text.length() + " characters");
							else
								alert("Info", "Your clipboard contains " + text);
					}



				clipboard_set_raw_text:

					Sets the text on the user's clipboard, using the system encoding.

					bool clipboard_set_raw_text(const string&in text);

				Arguments:

					* const string&in text: the text to copy, assumed to be in the system's encoding.

				Returns:

					Bool: true on success, false on failure.

				Remarks:

					To copy UTF-8 text, see clipboard_set_text().

				Example:

					```NVGT
					void main() {
						string text = input_box("Text", "Enter the text to copy.");
						clipboard_set_raw_text(text);
						if (text == "")
							alert("Info", "Your clipboard has been cleared.");
						else
							alert("Info", "Text copied");
					}



				clipboard_set_text:

					Sets the text on the user's clipboard.

					bool clipboard_set_text(const string&in text);

				Arguments:

					* const string&in text: the text to copy, assumed to be UTF_8.

				Returns:

					Bool: true on success, false on failure.

				Remarks:

					To copy text in the system encoding, see clipboard_set_raw_text().

				Example:

					```NVGT
					void main() {
						string text = input_box("Text", "Enter the text to copy.");
						clipboard_set_text(text);
						if (text == "")
							alert("Info", "Your clipboard has been cleared.");
						else
							alert("Info", "Text copied");
					}



				destroy_window:

					Destroys the currently shown window, if it exists.

					bool destroy_window();

				Returns:

					bool: true if the window was successfully destroyed, false otherwise.

				Example:

					```NVGT
					void main() {
						alert("Info", "Pressing SPACE will destroy the window");
						show_window("Example");
						while (true) {
							wait(5); // Don't hog all the CPU time.
							if (key_pressed(KEY_SPACE)) {
								destroy_window();
								wait(2000); // Simulate doing things while the window isn't active.
								exit();
							}
						}
					}



				exit:

					Completely shut down your application, returning a particular exit code to your operating system in the process.

					void exit(int exit_code = 0);

				Arguments:

					* int exit_code = 0: the exit code to return to your operating system.

				Example:

					```NVGT
					void main() {
						show_window("Example");
						screen_reader_output("Press escape to exit.", true);
						while (true) {
							wait(5);
							if (key_pressed(KEY_ESCAPE)) {
								screen_reader_output("Goodbye.", true);
								wait(500);
								exit();
							}
						}
					}



				focus_window:

					Attempt to bring your NVGT window to the foreground.

					bool focus_window();

				Returns:

					bool: true if the window was successfully focused, false otherwise.

				Example:

					```NVGT
					void main() {
						timer focuser;
						show_window("Focus every 5 seconds example");
						while (true) {
							wait(5);
							if (key_pressed(KEY_ESCAPE)) exit();
							if (focuser.has_elapsed(5000)) {
								focuser.restart();
								focus_window();
							}
						}
					}



				get_characters:

					Determine any printable characters typed into the games window since this function's lass call.

					string get_characters();

				Returns:

					string: Any characters typed since this function was last called, or an empty string if none are available.

				Remarks:

					This is the function one would use if they wished to integrate a virtual text field of any sort into their games or applications.

					Unlike functions such as key_down or keys_pressed, this function focuses specifically on textual content which has been input to the application. A character returned from this function, for example, could be anything from the space bar to an emoticon inserted using the windows emoji picker.

					A typical use case would involve calling this function once per iteration of your game loop and storing any new characters that are typed into a buffer that composes a chat message or a username or anything else requiring virtual text input.

				Example:

					```NVGT
					void main() {
						string buffer;
						show_window("get_characters example");
						while (!key_pressed(KEY_ESCAPE)) {
							wait(5);
							string char = get_characters();
							if (!char.empty()) {
								buffer += char;
								screen_reader_speak(char, true); // You need to process this further for screen readers, such as replacing . with period and other punctuation.
							}
							if (key_pressed(KEY_RETURN)) {
								screen_reader_speak("You typed " + (!buffer.empty()? buffer : "nothing"), true);
								buffer = "";
							}
						}
					}



				get_touch_device_name:
					Determine the name of a touch device given it's ID.

					`string get_touch_device_name(uint64 device_id);`

				Returns:
					string : The name of the given device, or an empty string on failure.


				get_touch_device_type:
					Determine the type of a touch device given it's ID.

					`touch_device_type get_touch_device_name(uint64 device_id);`

				Returns:
					touch_device_type: The type of the given device, or TOUCH_DEVICE_TYPE_INVALID if a bad ID is given.


				get_touch_devices:

					Enumerate all available touch input devices on the system.

					uint64[]@ get_touch_devices();

				Returns:

					uint64[]@: An array of touch device IDs, may be empty of no touch devices are available.

				Remarks:

					You can pass any of the IDs returned from this function to the get_touch_device_type, get_touch_device_name, or query_touch_device functions.

					Note that on some platforms, certain touch devices might not become available until they first receive input. For example this is true with the Android touch screen.

				Example:

					```NVGT
					void main() {
						uint64[]@ devices = get_touch_devices();
						if (devices.length() < 1) {
							alert("no devices", "There are no detectable touch devices on your system.");
							exit();
						}
						for (uint i = 0; i < devices.length(); i++)
							alert("device " + devices[i], get_touch_device_name(devices[i]));
					}



				get_window_height:

					Gets the height of the game window in pixels.

					int get_window_height();

				Returns:

					int: The height of the window in pixels, or 0 if no window is currently open.

				Example:

					```NVGT
					void main() {
						show_window("Window Size Example");
						wait(50); // Give the window time to appear
						int width = get_window_width();
						int height = get_window_height();
						alert("Window Size", "The window is " + width + " pixels wide and " + height + " pixels high.");
					}



				get_window_os_handle:

					Returns the native handle used by the operating system to manage NVGT's window.

					uint64 get_window_os_handle();

				Returns:

					uint64: An HWND / NSWindow\* / similar handle that the operating system uses to manage the window.

				Remarks:

					This function is only useful for someone trying to do something advanced, usually involving passing the identifier for the game window to a system API function or something similar. If you don't understand what this function does, you should probably avoid it for the present.

				Example:

					```NVGT
					void main() {
						// Lets set the window title using the windows API, typically you'd use show_window a second time for this.
						show_window("hello");
						uint64 handle = get_window_os_handle();
						library lib;
						lib.load("user32");
						lib.call("int SetWindowTextA(uint64, ptr)", handle, "example window");
						wait(500); // Let the user observe the change before exiting.
					}



				get_window_text:

					Returns the title of your window.

					string get_window_text();

				Returns:

					string: the title of your window.

				Example:

					```NVGT
					void main() {
						string[] possible_titles = {"Hello world", "NVGT test", "My game", "Example", "Window title example"};
						show_window(possible_titles[random(0, possible_titles.length() - 1)]);
						while (true) {
							wait(5);
							if (key_pressed(KEY_ESCAPE)) exit();
							if (key_pressed(KEY_SPACE)) {
								screen_reader_output("The window title currently is: " + get_window_text(), true);
								show_window(possible_titles[random(0, possible_titles.length() - 1)]); // Randomize the title again.
							}
						}
					}



				get_window_width:

					Gets the width of the game window in pixels.

					int get_window_width();

				Returns:

					int: The width of the window in pixels, or 0 if no window is currently open.

				Example:

					```NVGT
					void main() {
						show_window("Window Size Example");
						wait(50); // Give the window time to appear
						int width = get_window_width();
						int height = get_window_height();
						alert("Window Size", "The window is " + width + " pixels wide and " + height + " pixels high.");
					}



				idle_ticks:

					Obtain the time passed in milliseconds since a user has pressed a key or used the mouse.

					uint64 idle_ticks();

				returns:

					uint64: the number of milliseconds since a user has last interacted with the machine.

				remarks:

					This function can serve as a useful utility to verify whether a user has been away from the keyboard.

					It is currently integrated for windows and MacOS. On linux, it will always return 0 until further notice.

				Example:

					```NVGT
					void main() {
						int64 t;
						show_window("Idle test");
						wait(50);
						screen_reader_speak("Press space to check the idle time and escape to exit.", true);
						while (!key_pressed(KEY_ESCAPE)) {
							wait(5);
							if (key_pressed(KEY_SPACE))
								screen_reader_speak("You have been idle for %0 milliseconds".format(t), true);
							t = idle_ticks();
						}
					}



				info_box:

					Displays a dialog to the user consisting of a read-only text field and a close button.

					bool info_box(const string&in title, const string&in label, const string&in text, uint64 flags = 0);

				Arguments:

					* const string&in title: the title of the dialog.

					* const string&in label: the text to display above the text field.

					* const string&in text: the text to display in the text field.

					* uint64 flags = 0: a combination of flags, see message_box_flags for more information.

				Returns:

					bool: true if the user pressed the close button, false otherwise.

				Example:

					```NVGT
					void main() {
						info_box("Test", "Information", "this is a\r\nlong string\r\nthat can be split\r\nacross lines");
					}



				input_box:

					Displays a dialog to the user consisting of an input field, an OK button and a cancel button.

					string input_box(const string&in title, const string&in text, const string&in default_text = "", uint64 flags = 0);

				Arguments:

					* const string&in title: the title of the input box.

					* const string&in text: the text to display above the text field.

					* const string&in default_text = "": the contents to populate the text box with by default.

					* uint64 flags = 0: a combination of flags, see message_box_flags for more information.

				Returns:

					string: the text that the user typed.

				Example:

					```NVGT
					void main() {
						string name = input_box("Name", "What is your name?", "John Doe");
						if (name.is_empty()) {
							alert("Error", "You didn't type a name, unidentified person!");
							exit();
						}
						alert("Hi", name + "!");
					}



				install_keyhook:

					Attempts to install NVGT's JAWS keyhook.

					bool install_keyhook(bool allow_reinstall = true);

				Arguments:

					* bool allow_reinstall = true: whether or not this function will reinstall the hook on subsequent calls.

				Returns:

					bool: true if the keyhook was successfully installed, false otherwise.

				Remarks:

					This keyhook allows NVGT games to properly capture keyboard input while the JAWS for Windows screen reader is running.

					Note that installing it when JAWS is not running may yield unintended consequences, such as functionality of other screen readers being slightly limited while in your game window.

				Example:

					```NVGT
					void main() {
						bool success = install_keyhook();
						if (success) {
							alert("Info", "The keyhook was successfully installed!");
							uninstall_keyhook();
						}
						else
							alert("Info", "The keyhook was not successfully installed.");
					}



				is_console_available:

					Determine if your application has access to a console window or not.

					bool is_console_available();

				Returns:

					bool: true if your application has access to a console window, false otherwise.

				Remarks:

					If this function returns false, it is very likely that any output passed to the print or println functions or the cout/cerr datastreams will be silently dropped.

				Example:

					```NVGT
					void main() {
						if (is_console_available()) println("hello from the console!");
						else alert("console unavailable", "hello from the user interface!");
					}



				is_window_active:

					Determines if your game's window is active (i.e. has the keyboard focus).

					bool is_window_active();

				Returns:

					bool: true if your game window has the keyboard focus, false if not.

				Example:

					```NVGT
					void main() {
						show_window("Test");
						wait(1000); // Give the user time to alt+tab away if they so choose.
						bool active = is_window_active();
						if (active)
							alert("Info", "Your game window is active.");
						else
							alert("Info", "Your game window is not active.");
					}



				is_window_hidden:

					Determines if your game's window is hidden or not.

					bool is_window_hidden();

				Returns:

					bool: true if your game window is hidden, false if it's shown.

				Example:

					```NVGT
					void main() {
						show_window("Test");
						bool hide = random_bool();
						if (hide) hide_window();
						alert("The window is", is_window_hidden() ? "hidden" : "shown");
					}



				key_down:

					Determine if a particular key is held down.

					bool key_down(uint key);

				Arguments:

					* uint key: the key to check.

				Returns:

					bool: true if the key is down at all, false otherwise.

				Remarks:

					For a complete list of keys that can be passed to this function, see input constants.

				Example:

					```NVGT
					void main() {
						show_window("Example");
						while (true) {
							wait(5);
							if (key_down(KEY_SPACE))
								screen_reader_output("space", true);
							if (key_pressed(KEY_ESCAPE))
								exit();
						}
					}



				key_pressed:

					Determine if a particular key is pressed.

					bool key_pressed(uint key);

				Arguments:

					* uint key: the key to check.

				Returns:

					bool: true if the key was just pressed, false otherwise.

				Remarks:

					For a complete list of keys that can be passed to this function, see input constants.

				Example:

					```NVGT
					void main() {
						show_window("Example");
						while (true) {
							wait(5);
							if (key_pressed(KEY_SPACE))
								screen_reader_output("You just pressed space!", true);
							if (key_pressed(KEY_ESCAPE))
								exit();
						}
					}



				key_released:

					Determine if a particular key was just released.

					bool key_released(uint key);

				Arguments:

					* uint key: the key to check.

				Returns:

					bool: true if the key was just released, false otherwise.

				Remarks:

					For a complete list of keys that can be passed to this function, see input constants.

				Example:

					```NVGT
					void main() {
						show_window("Example");
						while (true) {
							wait(5);
							if (key_released(KEY_SPACE))
								screen_reader_output("you just released the space key", true);
							if (key_pressed(KEY_ESCAPE))
								exit();
						}
					}



				key_repeating:

					Determine if a particular key is repeating (i.e. it's being held, but wasn't just pressed).

					bool key_repeating(uint key);

				Arguments:

					* uint key: the key to check.

				Returns:

					bool: true if the key is repeating, false otherwise.

				Remarks:

					For a complete list of keys that can be passed to this function, see input constants.

				Example:

					```NVGT
					void main() {
						show_window("Example");
						while (true) {
							wait(5);
							if (key_repeating(KEY_SPACE))
								screen_reader_output("space", true);
							if (key_pressed(KEY_ESCAPE))
								exit();
						}
					}



				key_up:

					Determine if a particular key is up.

					bool key_up(uint key);

				Arguments:

					* uint key: the key to check.

				Returns:

					bool: true if the key is up, false otherwise.

				Remarks:

					For a complete list of keys that can be passed to this function, see input constants.

				Example:

					```NVGT
					void main() {
						int last_spoken = ticks(); // Speak periodically to avoid overwhelming the user (or their screen reader).
						show_window("Example");
						while (true) {
							wait(5);
							if (key_up(KEY_SPACE) && (ticks() - last_spoken) >= 500) {
								last_spoken = ticks();
								screen_reader_output("space is up", true);
							}
							if (key_pressed(KEY_ESCAPE))
								exit();
						}
					}



				keys_down:
					Returns a handle to an array of keys that are held down.

					`uint[]@ keys_down();`

				Returns:
					uint[]@: a handle to an array containing keys that are held down.

				Remarks:
					For a complete list of keys that can be returned by this function, see input constants.


				keys_pressed:
					Returns a handle to an array of keys that were just pressed.

					`uint[]@ keys_pressed();`

				Returns:
					uint[]@: a handle to an array containing keycodes that were just pressed.

				Remarks:
					For a complete list of keys that can be returned by this function, see input constants.


				keys_released:
					Returns a handle to an array of keys that were just released.

					`uint[]@ keys_released();`

				Returns:
					uint[]@: a handle to an array containing keycodes that were just released.

				Remarks:
					For a complete list of keys that can be returned by this function, see input constants.


				message_box:

					Displays a customizable message box.

					int message_box(const string&in title, const string&in text, string[]@ buttons, uint flags = 0);

				Arguments:

					* const string&in title: the title of the message box.

					* const string&in text: the text of the message box.

					* string[]@ buttons: a string array of button names, see remarks for more info.

					* uint flags = 0: a combination of flags (see message_box_flags for more information).

				Returns:

					int: the number of the button that was pressed, according to its position in the buttons array.

				Remarks:

					Notes on button syntax:

					* A grave character (`) prepending button text is default enter key.

					* A tilde character (~) before text means default cancel.

				Example:

					```NVGT
					void main() {
						message_box("Hello there", "I am a message box with two buttons", {"`OK", "~Cancel"});
					}



				open_file_dialog:

					Displays a dialog from which a user may select a file to open.

					string open_file_dialog(string filters = "", string default_location = "")

				Arguments:

					* string filters = "": A list of file extension filters (see remarks).

					* string default_location = "": The absolute initial directory to select files from (empty for OS default).

				Returns:

					string: The path to a file or an empty string on failure/cancelation.

				Remarks:

					The file extension filters are specified in the format description:extension|description:extension|description:extension1;extension2;extension3... You can also use the filter \* as an all files filter. for example: audio files:mp3;wav;ogg|text files:txt;rtf;md|all files:* would produce 3 items in the file type combo box, those being audio files, text files and all files.

					On some platforms it might be possible for the user to select a file that does not yet exist, so you should check for that condition yourself after the user selects a file just to be safe if that is important to you.

					This function is blocking, though we plan to add a version of the API in the future that uses a callback function or an event you can wait on if you wish to use this dialog asynchronously.

					You should only call this function on your application's main thread, which is what happens by default unless you use any of NVGT's threading facilities.

				Example:

					```NVGT
					void main() {
						string location = open_file_dialog("NVGT scripts:nvgt", cwdir());
						if (location.empty()) return;
						alert("example", "you selected " + location);
					}



				query_touch_device:

					Check what fingers are in contact with a touch device and where they are on the screen, if any.

					touch_finger[]@ query_touch_device(uint64 device_id = 0);

				Arguments:

					* uint64 device_id = 0: The device to query, 0 for the last device that received input.

				Returns:

					touch_finger[]@: An array of structures containing touch coordinates and pressure (see remarks).

				Remarks:

					Each structure in the returned array represents a finger, thus the length of the array indicates how many fingers are touching the device.

					Touch_finger objects contain only 4 properties. Int64 id is a unique identifier of the finger that will remain the same as long as the finger touches the devices, float x and float y contain the coordinates of the finger on the device, and the float pressure property indicates how much force or pressure the finger is being applied to the device with.

					Usage of this structure other than to receive touch input is not endorsed by the NVGT developers and may lead to unexpected results, most notably creating a touch_finger instance manually will likely set it's properties to random uninitialized memory on the stack. You have been warned.

				Example:

					```NVGT
					void main() {
						// Report the number of fingers touching the device.
						show_window("touch example");
						tts_voice tts;
						int prev_finger_count = 0;
						tts.speak("ready", true);
						while (!key_pressed(KEY_ESCAPE) and !key_pressed(KEY_AC_BACK)) {
							wait(5);
							touch_finger[]@ fingers = query_touch_device(); // This will query the last device that was touched.
							if (fingers.length() != prev_finger_count) {
								tts.speak(fingers.length() + (fingers.length() != 1? " fingers" : " finger"), true);
								prev_finger_count = fingers.length();
							}
						}
					}



				question:

					Display a yes/no dialog box to the user.

					int question(const string&in title, const string&in text, bool can_cancel = false, uint flags = 0);

				Arguments:

					* const string&in title: the title of the dialog.

					* const string&in text: the text of the dialog.

					* bool can_cancel = false: determines if a cancel button is present alongside the other buttons.

					* uint flags = 0: a combination of flags (see message_box_flags for more information).

				Returns:

					int: the number of the button that was pressed (1 for yes, 2 for no).

				Example:

					```NVGT
					void main() {
						int res = question("Hello there", "Do you like pizza?");
						alert("Info", "You clicked " + (res == 1 ? "yes" : "no"));
					}



				refresh_window:
					Updates the game window, pulling it for any new events and responding to messages from the operating system.

					`void refresh_window();`

				Remarks:
					You do not need to call this function yourself so long as you use the recommended wait() function in your game's loops, as wait() implicitly calls this function. We provide refresh_window() standalone unless you want to use your own / a different sleeping mechanism such as nanosleep, in which case you do need to manually pull the window using this function.

					For UI windows to function on the vast mejority of operating systems, they must maintain a constant stream of communication between the operating system and the program that created them. If a window stops receiving and handling such communications for even a few seconds, it will first become laggy before the operating system decides that the window has gone dead and begins reporting that the application is in a not responding or busy state.

					To solve this problem in NVGT applications, refresh_window() is provided to do all of this message handling for you in one repeated function call. This function will pull the operating system for any new messages or events and process them, updating the states of functions like key_pressed, get_characters etc. If it is not repeatedly called in some way, your entire application will be disfunctional.


				save_file_dialog:

					Displays a dialog from which a user may select a location to save a file.

					string save_file_dialog(string filters = "", string default_location = "")

				Arguments:

					* string filters = "": A list of file extension filters (see remarks).

					* string default_location = "": The absolute initial directory to select from (empty for OS default).

				Returns:

					string: The path to save to or an empty string on failure/cancelation.

				Remarks:

					The file extension filters are specified in the format description:extension|description:extension|description:extension1;extension2;extension3... You can also use the filter \* as an all files filter. for example: audio files:mp3;wav;ogg|text files:txt;rtf;md|all files:* would produce 3 items in the file type combo box, those being audio files, text files and all files.

					Keep in mind that the user may select a file that already exists if they wish to overwrite it.

					This function is blocking, though we plan to add a version of the API in the future that uses a callback function or an event you can wait on if you wish to use this dialog asynchronously.

					You should only call this function on your application's main thread, which is what happens by default unless you use any of NVGT's threading facilities.

				Example:

					```NVGT
					void main() {
						string location = save_file_dialog("text files (*.txt):txt|all files:*");
						if (location.empty()) return;
						alert("example", "you selected " + location);
					}



				select_folder_dialog:

					Displays a dialog from which a user may select a folder.

					string select_folder_dialog(string default_location = "")

				Arguments:

					* string default_location = "": The absolute initial directory to select from (empty for OS default).

				Returns:

					string: The path to a folder or an empty string on failure/cancelation.

				Remarks:

					Be aware that on some platforms, the user might be able to select a directory that does not yet exist.

					This function is blocking, though we plan to add a version of the API in the future that uses a callback function or an event you can wait on if you wish to use this dialog asynchronously.

					You should only call this function on your application's main thread, which is what happens by default unless you use any of NVGT's threading facilities.

				Example:

					```NVGT
					void main() {
						string location = select_folder_dialog();
						if (location.empty()) return;
						alert("example", "you selected " + location);
					}



				set_application_name:
					This function lets you set the name of your application. This is a name that's sent to your operating system in certain instances, for example in volume mixer dialogs or when an app goes not responding. It does not effect the window title or anything similar.

					`bool set_application_name(const string&in application_name);`

				Arguments:
					* const string&in application_name: the name to set for your application.

				Returns:
					bool: true if the name was successfully set, false otherwise.


				show_window:

					Shows a window with the specified title.

					bool show_window(const string&in title, uint flags = 0);

				Arguments:

					* const string&in title: the title of the window.

					* uint flags: An optional set of flags to control the window's appearance. This can be a combination of values from the `window_flags` enum.

				Returns:

					bool: true if window creation was successful, false otherwise.

				Remarks:

					This window doesn't do any event processing by default. As such, you have to create a loop in order to keep it alive.

				Example:

					```NVGT
					void main() {
						alert("Info", "Press escape to close the window.");
						show_window("Example", WINDOW_FLAG_RESIZABLE | WINDOW_FLAG_MAXIMIZED);
						while (true) {
							wait(5); // Don't hog all the CPU.
							if (key_pressed(KEY_ESCAPE))
								exit();
						}
					}



				total_keys_down:

					Returns the total number of keys that are currently held down.

					int total_keys_down();

				Returns:

					int: the number of keys currently held down.

				Example:

					```NVGT
					void main() {
						int check_time = 1000, orig_ticks = ticks();
						show_window("Example");
						while (true) {
							wait(5);
							if ((ticks() - orig_ticks) >= check_time) {
								orig_ticks = ticks();
								int key_count  = total_keys_down();
								screen_reader_output(key_count + " " + (key_count == 1 ? "key is" : "keys are") + " currently held down.", true);
							}
							if (key_pressed(KEY_ESCAPE))
								exit();
						}
					}



				uninstall_keyhook:

					Uninstall's NVGT's JAWS keyhook.

					void uninstall_keyhook();

				Remarks:

					This keyhook allows NVGT games to properly capture keyboard input while the JAWS for Windows screen reader is running.

				Example:

					```NVGT
					void main() {
						install_keyhook();
						alert("Info", "Keyhook installed. Uninstalling...");
						uninstall_keyhook();
						alert("Done", "Keyhook uninstalled.");
					}



				urlopen:

					Opens the specified URL in the appropriate application, for example an https:// link in your web browser, or a tt://link in TeamTalk.

					bool urlopen(const string&in url);

				Arguments:

					* const string&in url: The URL to open.

				Returns:

					bool: true if the URl was successfully opened, false otherwise.

				Remarks:

					What application this function opens depends on what the user has set on their system; there's no way for you to control it.

				Example:

					```NVGT
					void main() {
						urlopen("https://nvgt.dev");
					}



				wait:

					Waits for a specified number of milliseconds.

					void wait(int milliseconds);

				Arguments:

					* int milliseconds: the number of milliseconds to wait for.

				Remarks:

					This function blocks the calling thread, meaning no other work can happen on that thread until the wait period is over.

					On the main thread, it performs other essential tasks, such as incremental garbage collection, internal event processing, and refreshing the window.

					It is strongly advised to always put `wait(5);` in your main loops. This prevents your game from hogging the CPU, and also makes sure that the window remains responsive and has been updated with the latest input events from the user.

					Please note that, due to low-level operating system scheduling behaviors, the millisecond parameter is not an exact value, but rather a guaranteed minimum. This is even more important to remember when calling from the main thread given the additional tasks performed. It should therefore never be used as a time marker where accuracy is required.

				Example:

					```NVGT
					void main() {
						alert("Info", "Once you press OK, I will wait for 2000 milliseconds (2 seconds).");
						wait(2000);
						alert("Info", "2000 milliseconds passed!");
					}



	Bundled Includes API:
		Auditory User Interface (form.nvgt):
			Classes:
				audio_form:
					This class facilitates the easy creation of user interfaces that convey their usage entirely through audio.

				Notes:
					* many of the methods in this class only work on certain types of controls, and will return false and set an error value if used on invalid types of controls. This will generally be indicated in the documentation for each function.
					* Exceptions are not used here. Instead, we indicate errors through `audio_form::get_last_error()`.
					* An audio form object can have up to 50 controls.


					Methods:
						activate_progress_timer:
							Activate progress updates on a progress bar control.

							`bool audio_form::activate_progress_timer(int control_index);`

						Arguments:
							* int control_index: the index of the progress bar.

						Returns:
							bool: true if the progress bar was successfully activated, false otherwise.


						add_list_item:
							Add a string item to a list control.

							1. `bool audio_form::add_list_item(int control_index, string option, int position = -1, bool selected = false, bool focus_if_first = true);`
							2. `bool audio_form::add_list_item(int control_index, string option, string id, int position = -1, bool selected = false, bool focus_if_first = true);`

						Arguments (1):
							* int control_index: the control index of the list to add to.
							* string option: the item to add to the list.
							* int position = -1: the position to insert the new item at (-1 = end of list).
							* bool selected = false: should this item be selected by default?
							* bool focus_if_first = true: if this item is the first in the list and no other item gets explicitly focused, this item will be focused.

						Arguments (2):
							* int control_index: the control index of the list to add to.
							* string option: the item to add to the list.
							* string id: the ID of the item in the list.
							* int position = -1: the position to insert the new item at (-1 = end of list).
							* bool selected = false: should this item be selected by default?
							* bool focus_if_first = true: if this item is the first in the list and no other item gets explicitly focused, this item will be focused.

						Returns:
							bool: true if the item was successfully added, false otherwise.

						Remarks:
							This function only works on list controls.


						clear_list:
							Clear a list control of all its items.

							`bool audio_form::clear_list(int control_index);`

						Arguments:
							* int control_index: the index of the list to clear.

						Returns:
							bool: true if the list was successfully cleared, false otherwise.


						create_button:
							Creates a new button and adds it to the audio form.

							`int audio_form::create_button(string caption, bool primary = false, bool cancel = false, bool overwrite = true);`

						Arguments:
							* string caption: the label to associate with the button.
							* bool primary = false: should this button be activated by pressing enter anywhere in the form?
							* bool cancel = false: should this button be activated by pressing escape anywhere in the form?
							* bool overwrite = true: overwrite any existing primary/cancel settings.

						Returns:
							int: the control index of the new button, or -1 if there was an error. To get error information, look at `audio_form::get_last_error();`.


						create_checkbox:
							Creates a new checkbox and adds it to the audio form.

							`int audio_form::create_checkbox(string caption, bool initial_value = false, bool read_only = false);`

						Arguments:
							* string caption: the text to be read when tabbing over this checkbox.
							*( bool initial_value = false: the initial value of the checkbox (true = checked, false = unchecked).
							* bool read_only = false: can the user check/uncheck this checkbox?

						Returns:
							int: the control index of the new checkbox, or -1 if there was an error. To get error information, see `audio_form::get_last_error();`.


						create_input_box:
							Creates an input box control on the audio form.

							`int audio_form::create_input_box(string caption, string default_text = "", string password_mask = "", int maximum_length = 0, bool read_only = false, bool multiline = false, bool multiline_enter = true);`

						Arguments:
							* string caption: the label of the input box (e.g. what will be read when you tab over it?).
							* string default_text = "": the text to populate the input box with by default (if any).
							* string password_mask = "": a string to mask typed characters with, (e.g. "star"). Mainly useful if you want your field to be password protected. Leave blank for no password protection.
							* int maximum_length = 0: the maximum number of characters that can be typed in this field, 0 for unlimited.
							* bool read_only = false: should this text field be read-only?
							* bool multiline = false: should this text field have multiple lines?
							* bool multiline_enter = true: should pressing enter in this field insert a new line (if it's multiline)?

						Returns:
							int: the control index of the new input box, or -1 if there was an error. To get error information, look at `audio_form::get_last_error();`.


						create_keyboard_area:
							Creates a new keyboard area and adds it to the audio form.

							`int audio_form::create_keyboard_area(string caption);`

						Arguments:
							* string caption: the text to be read when tabbing onto the keyboard area.

						Returns:
							int: the control index of the new keyboard area, or -1 if there was an error. To get error information, see `audio_form::get_last_error();`.


						create_link:
							Creates a new link and adds it to the audio form.

							`int audio_form::create_link(string caption, string url);`

						Arguments:
							* string caption: the label of the link.
							* string url: The link to open.

						Returns:
							int: the control index of the new link, or -1 if there was an error. To get error information, see `audio_form::get_last_error();`.

						Remarks:
							The link control is similar to buttons, but this opens the given link when pressed. This also speaks the error message if the link cannot be opened or the link isn't an actual URL. If you want the maximum possible customization, use `audio_form::create_button` method instead.


						create_list:
							Creates a new list control and adds it to the audio form.

							`int audio_form::create_list(string caption, int maximum_items = 0, bool multiselect = false, bool repeat_boundary_items = false);`

						Arguments:
							* string caption: the label to attach to this list.
							* int maximum_items = 0: the maximum number of allowed items, 0 for unlimited.
							* bool multiselect = false: can the user select multiple items in this list?
							* bool repeat_boundary_items = false: do items repeat if you press the arrow keys at the edge of the list?

						Returns:
							int: the control index of the new list, or -1 if there was an error. To get error information, look at `audio_form::get_last_error();`.


						create_progress_bar:
							Creates a new progress bar control and adds it to the audio form.

							`int audio_form::create_progress_bar(string caption, int speak_interval = 5, bool speak_global = true);`

						Arguments:
							* string caption: the label to associate with the progress bar.
							* int speak_interval = 5: how often to speak percentage changes.
							* bool speak_global = true: should progress updates be spoken even when this control doesn't have keyboard focus?

						Returns:
							int: the control index of the new progress bar, or -1 if there was an error. To get error information, look at `audio_form::get_last_error();`.


						create_slider:
							Creates a new slider control and adds it to the audio form.

							`int audio_form::create_slider(string caption, double default_value = 50, double minimum_value = 0, double maximum_value = 100, string text = "", double step_size = 1);`

						Arguments:
							* string caption: the text to be spoken when this slider is tabbed over.
							* double default_value = 50: the default value to set the slider to.
							* double minimum_value = 0: the minimum value of the slider.
							* double maximum_value = 100: the maximum value of the slider.
							* string text = "": extra text to be associated with the slider.
							* double step_size = 1: the value that will increment/decrement the slider value.

						Returns:
							int: the control index of the new slider, or -1 if there was an error. To get error information, see `audio_form::get_last_error();`.


						create_status_bar:
							Creates a new status bar and adds it to the audio form.

							`int audio_form::create_status_bar(string caption, string text);`

						Arguments:
							* string caption: the label of the status bar.
							* string text: the text to display on the status bar.

						Returns:
							int: the control index of the new status bar, or -1 if there was an error. To get error information, see `audio_form::get_last_error();`.


						create_subform:

							Creates a new sub form and adds it to the audio form.

							int audio_form::create_subform(string caption, audio_form@ f);

						Arguments:

							* string caption: the label to associate with the sub form.

							* audio_form@ f: an object pointing to an already created form.

						Returns:

							int: the control index of the new sub form control, or -1 if there was an error. To get error information, look at `audio_form::get_last_error();`.

						Example:

							```NVGT
							#include "form.nvgt"
							// some imaginary global application variables.
							bool logostart = true, menuwrap = false, firespace = false, radar = true;
							// First, lets make a class which stores a category name and the form that the category is linked to.
							class settings_category {
								string name;
								audio_form@ f;
								settings_category(string n, audio_form@ f) {
									this.name = n;
									@this.f = @f;
								}
							}
							void settings_dialog() {
								// Define some categories and settings on each category like this:
								audio_form fc_general;
								fc_general.create_window();
								int f_logostart = fc_general.create_checkbox("Play &logo on startup", logostart);
								int f_menuwrap = fc_general.create_checkbox("Wrap &menus", menuwrap);
								audio_form fc_gameplay;
								fc_gameplay.create_window();
								int f_firespace = fc_gameplay.create_checkbox("Use space instead of control to &fire", firespace);
								int f_radar = fc_gameplay.create_checkbox("Enable the &radar", firespace);
								// Add as many categories as you want this way.
								audio_form f; // The overarching main form.
								f.create_window("Settings", false, true);
								int f_category_list = f.create_tab_panel("&Category"); // The user will select categories from here. Note: you can also use create_list.
								int f_category_display = f.create_subform("General settings", @fc_general); // Now by default, the main form embeds the general category form right there.
								int f_ok = f.create_button("&Save settings", true);
								int f_cancel = f.create_button("Cancel", false, true);
								// Now lets create a structured list of categories that can be browsed based on the class above.
								settings_category@[] categories = {
									settings_category("General", @fc_general),
									settings_category("Gameplay", @fc_gameplay)
								};
								// And then add the list of categories to the form's list.
								for (uint i = 0; i < categories.length(); i++) {
									f.add_list_item(f_category_list, categories[i].name);
								}
								// Focus the form's list position on the general category, then set the form's initial focused control to the category list.
								f.set_list_position(f_category_list, 0);
								f.focus(0);
								settings_category@ last_category = @categories[0]; // A handle to the currently selected category so we can detect changes to the selection.
								// Finally this is the loop that does the rest of the magic.
								while (!f.is_pressed(f_cancel)) {
									wait(5);
									f.monitor();
									int pos = f.get_list_position(f_category_list);
									settings_category@ selected_category = null;
									if (pos > -1 and pos < categories.length())
										@selected_category = @categories[pos];
									if (@last_category != @selected_category) {
										last_category.f.subform_control_index = -1; // Later improvements to audio form will make this line be handled internally.
										last_category.f.focus_silently(0); // Make sure that if the category is reselected, it is focused on the first control.
										@last_category = @selected_category;
										f.set_subform(f_category_display, @selected_category.f);
										f.set_caption(f_category_display, selected_category.name + " settings");
									}
									// The following is a special feature I came up with in stw which makes it so that if you are in the category list, keyboard shortcuts from the entire form will work regardless of category.
									if (f.get_current_focus() == f_category_list and key_down(KEY_LALT) or key_down(KEY_RALT)) {
										for (uint i = 0; i < categories.length(); i++) {
											if (categories[i].f.check_shortcuts(true)) {
												f.set_list_position(f_category_list, i);
												@last_category = @categories[i];
												f.set_subform(f_category_display, @last_category.f);
												f.set_caption(f_category_display, last_category.name + " settings");
												f.focus(f_category_display);
												break;
											}
										}
									}
									// Now we can finally check for the save button.
									if (f.is_pressed(f_ok)) {
										logostart = fc_general.is_checked(f_logostart);
										menuwrap = fc_general.is_checked(f_menuwrap);
										firespace = fc_gameplay.is_checked(f_firespace);
										radar = fc_gameplay.is_checked(f_radar);
										return;
									}
								}
							}
							// Lets make this thing run so we can see it work.
							void main() {
								show_window("test");
								settings_dialog();
							}



						create_window:

							Creates a window to show audio form controls.

							1. void audio_form::create_window();

							2. void audio_form::create_window(string window_title, bool change_screen_title = true, bool say_dialog = true, bool silent = false);

						Arguments (2):

							* string window_title: the title of the window.

							* bool change_screen_title = true: whether or not the main window's title should be set as well.

							* bool say_dialog = true: whether or not the window should be reported as a dialog (in the context of the audio form).

							* bool silent = false: should this window be shown silently?

						Example:

							```NVGT
							#include "form.nvgt"
							#include "speech.nvgt"
							void main() {
								audio_form f;
								f.create_window("Test");
								wait(50); // Give screen readers time to speak the window title.
								int f_exit = f.create_button("exit");
								f.focus(f_exit);
								while (true) {
									wait(5);
									f.monitor();
									if (f.is_pressed(f_exit))
										exit();
								}
							}



						delete_control:
							Removes a control with a particular index from the audio form.

							`bool audio_form::delete_control(int control_index);`

						Arguments:
							* int control_index: the index of the control to delete.

						Returns:
							bool: true if the control was successfully deleted, false otherwise.


						delete_list_item:
							Remove an item from a list control.

							`bool audio_form::delete_list_item(int control_index, int list_index, bool reset_cursor = true, bool speak_deletion_status = true);`

						Arguments:
							* int control_index: the index of the list to remove the item from.
							* int list_index: the index of the item to remove.
							* bool reset_cursor = true: should the user's cursor position be reset to the top of the list upon success?
							* bool speak_deletion_status = true: should the user be informed of the deletion via speech feedback?

						Returns:
							bool: true if the item was successfully deleted, false otherwise.


						delete_list_selections:
							Unselect any currently selected items in a list control.

							`bool audio_form::delete_list_selections(int control_index, bool reset_cursor = true, bool speak_deletion_status = true);`

						Arguments:
							* int control_index: the index of the list to unselect items in.
							* bool reset_cursor = true: should the user's cursor position be reset to the top of the list upon success?
							* bool speak_deletion_status = true: should the user be informed of the unselection via speech feedback?

						Returns:
							bool: true if the selection was successfully cleared, false otherwise.


						edit_list_item:
							Edits the value of a list item.

							`bool audio_form::edit_list_item(int control_index, string new_option, int position);`

						Arguments:
							* int control_index: the index of the list containing the item.
							* string new_option: the new text of the item.
							* int position: the item's index in the list.

						Returns:
							bool: true if the item's value was successfully updated, false otherwise.


						edit_list_item_id:
							Modifies the ID of a list item.

							`bool audio_form::edit_list_item_id(int control_index, string new_id, int position);`

						Arguments:
							* int control_index: the index of the list containing the item.
							* string new_id: the new ID of the list item.
							* int position: the item's index in the list.

						Returns:
							bool: true if the item's ID was successfully updated, false otherwise.


						focus:
							Set a particular control to have the keyboard focus, and notify the user.

							`bool audio_form::focus(int control_index);`

						Arguments:
							* int control_index: the index of the control to focus.

						Returns:
							bool: true if the control was successfully focused, false otherwise.


						focus_interrupt:
							Set a particular control to have the keyboard focus, and notify the user (cutting off any previous speech).

							`bool audio_form::focus_interrupt(int control_index);`

						Arguments:
							* int control_index: the index of the control to focus.

						Returns:
							bool: true if the control was successfully focused, false otherwise.


						focus_silently:
							Set a particular control to have the keyboard focus, without notifying the user.

							`bool audio_form::focus_silently(int control_index);`

						Arguments:
							* int control_index: the index of the control to focus.

						Returns:
							bool: true if the control was successfully focused, false otherwise.


						get_cancel_button:
							Get the control index of the cancel button (e.g. the button activated when pressing escape anywhere on an audio form).

							`int audio_form::get_cancel_button();`

						Returns:
							int: the control index of the cancel button.


						get_caption:
							Get the caption of a control.

							`string audio_form::get_caption(int control_index);`

						Arguments:
							* int control_index: the index of the control to query.

						Returns:
							string: the caption of the control.


						get_checked_list_items:
							Get a list of all currently checked items in a list control.

							`int[]@ audio_form::get_checked_list_items(int control_index);`

						Arguments:
							* int control_index: the index of the list to query.

						Returns:
							int[]@: handle to an array containing the index of every checked item in the list.

						Remarks:
							This function only works on multiselect lists. If you want something that also works on single-select lists, see `get_list_selections()`.


						get_control_count:
							Returns the number of controls currently present on the form.

							`int audio_form::get_control_count();`

						Returns:
							int: the number of controls currently on the form.


						get_control_type:
							Returns the type of a control.

							`int audio_form::get_control_type(int control_index);`

						Arguments:
							* int control_index: the index of the control to get the type of.

						Returns:
							int: the type of the control (see control_types for more information).


						get_current_focus:
							Get the control index of the control with the keyboard focus.

							`int audio_form::get_current_focus();`

						Returns:
							int: the control index that currently has the keyboard focus.


						get_custom_type:
							Get the custom type of the control, if available.

							`string audio_form::get_custom_type(int control_index);`

						Arguments:
							* int control_index: the control you want to check.

						Returns:
							string: the custom type set on this control if available, an empty string otherwise.


						get_default_button:
							Get the control index of the default button (e.g. the button activated when pressing enter anywhere on an audio form).

							`int audio_form::get_default_button();`

						Returns:
							int: the control index of the default button.


						get_last_error:
							Get the last error that was raised from this form.

							`int audio_form::get_last_error();`

						Returns:
							int: the last error code raised by this audio_form ( see audioform_errorcodes for more information).

						Remarks:
							As noted in the introduction to this class, exceptions are not used here. Instead, we indicate errors through this function.


						get_line_column:
							Get the current line column of an input box. 

							`int audio_form::get_line_column(int control_index);`

						Arguments:
							* int control_index: the index of the input box to retrieve the line column from.

						Returns:
							int: The line column of the input box, or -1 if there was an error.

						Remarks:
							This method only works on input boxes.


						get_line_number:
							Get the current line number of an input box. 

							`int audio_form::get_line_number(int control_index);`

						Arguments:
							* int control_index: the index of the input box to retrieve the line number from.

						Returns:
							int: The line number of the input box, or -1 if there was an error.

						Remarks:
							This method only works on input boxes.


						get_link_url:
							Retrieves the URL of the link control.

							`string audio_form::get_link_url(int control_index);`

						Arguments:
							* int control_index: the index of the control.

						Returns:
							string: the URL.

						Remarks:
							This method only works on link control.


						get_list_count:
							Get the number of items contained in a list control.

							`int audio_form::get_list_count(int control_index);`

						Arguments:
							* int control_index: index of the list control to query.

						Returns:
							int: the number of items in the list.


						get_list_index_by_id:
							Get the index of a list item by its ID.

							`int audio_form::get_list_index_by_id(int control_index, string id);`

						Arguments:
							* int control_index: the index of the list.
							* string id: the ID of the item to query.

						Returns:
							int: the index of the item, -1 on error.


						get_list_item:
							Get the text of a particular list item.

							`string audio_form::get_list_item(int control_index, int list_index);`

						Arguments:
							* int control_index: the index of the list.
							* int list_index: the index of the item to get.

						Returns:
							string: the text of the item.


						get_list_item_id:
							Get the ID of a particular list item.

							`string audio_form::get_list_item_id(int control_index, int list_index);`

						Arguments:
							* int control_index: the index of the list.
							* int list_index: the index of the item to get.

						Returns:
							string: the ID of the item.


						get_list_position:
							Get the user's currently focused item in a list control.

							`int audio_form::get_list_position(int control_index);`

						Arguments:
							* int control_index: the index of the list.

						Returns:
							int: the user's current cursor position in the list.


						get_list_selections:
							Get a list of all items currently selected in a list control.

							`int[]@ audio_form::get_list_selections(int control_index);`

						Arguments:
							* int control_index: the index of the list to query.

						Returns:
							int[]@: handle to an array containing the index of every selected item in the list (see remarks).

						Remarks:
							In the context of this function, a selected item is any item that  is checked (if the list supports multiselection), as well as the currently selected item. If you want to only get the state of checked list items, see the `get_checked_list_items()` function.


						get_progress:
							Get the value of a progress bar.

							`int audio_form::get_progress(int control_index);`

						Arguments:
							* int control_index: the index of the progress bar to query.

						Returns:
							int: the current value of the progress bar.

						Remarks:
							This method only works on progress bar controls.


						get_slider:
							Get the value of a slider.

							`double audio_form::get_slider(int control_index);`

						Arguments:
							* int control_index: the index of the slider to query.

						Returns:
							double: the current value of the slider. In case of error, this may return -1. To get error information, see `audio_form::get_last_error();`.

						Remarks:
							This method only works on slider control.


						get_slider_maximum_value:
							Get the maximum value of a slider.

							`double audio_form::get_slider_maximum_value(int control_index);`

						Arguments:
							* int control_index: the index of the slider to query.

						Returns:
							double: the maximum allowable value the slider may be set to. In case of error, this may return -1. To get error information, see `audio_form::get_last_error();`.

						Remarks:
							This method only works on slider control.


						get_slider_minimum_value:
							Get the minimum value of a slider.

							`double audio_form::get_slider_minimum_value(int control_index);`

						Arguments:
							* int control_index: the index of the slider to query.

						Returns:
							double: the minimum allowable value the slider may be set to. In case of error, this may return -1. To get error information, see `audio_form::get_last_error();`.

						Remarks:
							This method only works on slider control.


						get_text:
							Get the text from an input box or status bar.

							`string audio_form::get_text(int control_index);`

						Arguments:
							* int control_index: the index of the control to retrieve the text from.

						Returns:
							string: The text from the control, or an empty string if there was an error.

						Remarks:
							This method only works on input boxes and status bars.


						has_custom_type:
							Determines whether the control has its custom type set.

							`bool audio_form::has_custom_type(int control_index);`

						Arguments:
							* int control_index: the control you want to check.

						Returns:
							bool: true if the control has its custom type set, false otherwise.

						Remarks:
							Please note that this method is equivalent to `audio_form::get_custom_type(control_index).empty()`


						is_checked:
							Get the state of a checkbox.

							`bool audio_form::is_checked(int control_index);`

						Arguments:
							* int control_index: the index of the control to query.

						Returns:
							bool: true if the checkbox is checked, false otherwise.

						Remarks:
							This function only works on checkbox controls.


						is_disallowed_char:
							Determines whether the text of a given control contains characters that are not allowed, set by the `audio_form::set_disallowed_chars` method.

							`bool audio_form::is_disallowed_char(int control_index, string char, bool search_all = true);`

						Arguments:
							* int control_index: the index of the control.
							* string char: one or more characters to query
							* bool search_all = true: toggles whether to search character by character or to match the entire string.

						Returns:
							bool: true if the text of the control contains disallowed characters, false otherwise.

						Remarks:
							The `search_all` parameter will match the characters depending upon its state. If set to false, the entire string will be searched. If set to true, it will loop through each character and see if one of them contains disallowed character. Thus, you will usually set this to true. One example you might set to false is when the form only has 1 character length, but it is not required, it is looping each character already. However, it is also a good idea to turn this off for the maximum possible performance if you're sure that the input only requires 1 length of characters.


						is_enabled:
							Determine if a particular control is enabled or not.

							`bool audio_form::is_enabled(int control_index);`

						Arguments:
							* int control_index: the index of the control to query.

						Returns:
							bool: true if the control is enabled, false if it's disabled.


						is_list_item_checked:
							Determine if the list item at a particular index is checked or not.

							`bool audio_form::is_list_item_checked(int control_index, int item_index);`

						Arguments:
							* int control_index: the index of the list containing the item to be checked.
							* int item_index: the index of the item to check.

						Returns:
							bool: true if the item exists and is checked, false otherwise.


						is_multiline:
							Determine if a particular control is multiline or not.

							`bool audio_form::is_multiline(int control_index);`

						Arguments:
							* int control_index: the index of the control to query.

						Returns:
							bool: true if the control is multiline, false otherwise.

						Remarks:
							This function only works on input boxes.


						is_pressed:
							Determine if a particular button was just pressed.

							`bool audio_form::is_pressed(int control_index);`

						Arguments:
							* int control_index: the index of the control to query.

						Returns:
							bool: true if the button was just pressed, false otherwise.

						Remarks:
							This function only works on button controls.


						is_read_only:
							Determine if a particular control is read-only or not.

							`bool audio_form::is_read_only(int control_index);`

						Arguments:
							* int control_index: the index of the control to query.

						Returns:
							bool: true if the control is read-only, false if it's not.

						Remarks:
							This function only works on input boxes and checkboxes.


						is_visible:
							Determine if a particular control is visible or not.

							`bool audio_form::is_visible(int control_index);`

						Arguments:
							* int control_index: the index of the control to query.

						Returns:
							bool: true if the control is visible, false if it's invisible.


						monitor:
							Processes all keyboard input, and dispatches all events to the form. Should be called in your main loop.

							`int audio_form::monitor();`

						Returns:
							int: this value can be ignored for users of this class; it's used internally for subforms.


						pause_progress_timer:
							Pause updating of a progress bar control.

							`bool audio_form::pause_progress_timer(int control_index);`

						Arguments:
							* int control_index: the index of the progress bar.

						Returns:
							bool: true if the progress bar was paused, false otherwise.


						set_button_attributes:
							Set the primary and cancel flags of a button.

							`bool audio_form::set_button_attributes(int control_index, bool primary, bool cancel, bool overwrite = true);

						Arguments:
							* int control_index: the index of the button to update.
							* bool primary: should this button be made primary (e.g. pressing enter from anywhere on the form activates it)?
							* bool cancel: should this button be made the cancel button (e.g. should pressing escape from anywhere on the form always activate it?).
							* bool overwrite = true: should the previous primary and cancel buttons (if any) be overwritten?

						Returns:
							bool: true if the attributes were successfully set, false otherwise.


						set_caption:
							Sets the caption of a control.

							`bool audio_form::set_caption(int control_index, string caption);`

						Arguments:
							* int control_index: the index of the control.
							* string caption: the caption to set on the control (see remarks for more infromation).

						Returns:
							bool: true if the caption was successfully set, false otherwise.

						Remarks:
							The caption is read every time a user focuses this particular control.

							It is possible to associate hotkeys with controls by putting an "&" symbol in the caption. For example, setting the caption of a button to "E&xit" would assign the hotkey alt+x to focus it.


						set_checkbox_mark:
							Set the state of a checkbox (either checked or unchecked).

							`bool audio_form::set_checkbox_mark(int control_index, bool checked);`

						Arguments:
							* int control_index: the control index of the checkbox.
							* bool checked: whether the checkbox should be set to checked or not.

						Returns:
							bool: true if the operation was successful, false otherwise. To get error information, look at `audio_form::get_last_error();`.


						set_custom_type:
							Sets a custom type for a control.

							`bool audio_form::set_custom_type(int control_index, string custom_type);`

						Arguments:
							* int control_index: the index of the control.
							* string custom_type: a custom text type to set on the control.

						Returns:
							bool: true if the custom text type was successfully set, false otherwise.


						set_default_controls:
							Sets default and cancel buttons for this form.

							`bool set_default_controls(int primary, int cancel);`

						Arguments:
							* int primary: the control index of the button to make primary.
							* int cancel: the control index of the cancel button.

						Returns:
							bool: true if the controls were successfully set, false otherwise.


						set_default_keyboard_echo:
							Sets the default keyboard echo of controls on your form.

							`bool audio_form::set_default_keyboard_echo(int keyboard_echo, bool update_controls = true);`

						Arguments:
							* int keyboard_echo: the keyboard echo mode to use (see text_entry_speech_flags for more information).
							* bool update_controls = true: whether or not this echo should be applied to any controls that already exist on your form.

						Returns:
							bool: true if the echo was successfully set, false otherwise.


						set_disallowed_chars:
							Sets the whitelist/blacklist characters of a control.

							`bool audio_form::set_disallowed_chars(int control_index, string chars, bool use_only_disallowed_chars = false, string char_disallowed_description = "");`

						Arguments:
							* int control_index: the index of the control.
							* string chars: the characters to set.
							* bool use_only_disallowed_chars = false: sets whether the control should only use the characters in this list. true means use only characters that are in the list, and false means allow only characters that are not in the list.
							* string char_disallowed_description = "": the text to speak when an invalid character is inputted.

						Returns:
							bool: true if the characters were successfully set, false otherwise.

						Remarks:
							Setting the `use_only_disallowed_chars` parameter to true will restrict all characters that are not in the list. This is useful to prevent other characters in number inputs.

							Setting the chars parameter to empty will clear the characters and will switch back to default.

							Please note that these character sets will also restrict the text from being pasted if the clipboard contains disallowed characters.


						set_enable_go_to_index:
							Toggles whether the control can use go to line functionality.

							`bool audio_form::set_enable_go_to_index(int control_index, bool enabled);`

						Arguments:
							* int control_index: the index of the control.
							* bool enabled: enables the go to line functionality.

						Returns:
							bool: true if the state was successfully set, false otherwise.


						set_enable_search:
							Toggles whether the control can use search functionality.

							`bool audio_form::set_enable_search(int control_index, bool enabled);`

						Arguments:
							* int control_index: the index of the control.
							* bool enabled: enables the search functionality.

						Returns:
							bool: true if the state was successfully set, false otherwise.


						set_keyboard_echo:
							Set the keyboard echo for a particular control.

							`bool audio_form::set_keyboard_echo(int control_index, int keyboard_echo);`

						Arguments:
							* int control_index: the index of the control to modify.
							* int keyboard_echo: the keyboard echo mode to use (see text_entry_speech_flags for more information).

						Returns:
							bool: true if the keyboard echo was successfully set, false otherwise.


						set_link_url:
							Sets the URL of the link control.

							`bool audio_form::set_link_url(int control_index, string new_url);`

						Arguments:
							* int control_index: the index of the control.
							* string new_url: the URL to set.

						Returns:
							bool: true on success, false on failure.

						Remarks:
							This method only works on link control.


						set_list_multinavigation:
							Configures how the multi-letter navigation works in a list control.

							`bool audio_form::set_list_multinavigation(int control_index, bool letters, bool numbers, bool nav_translate = true);`

						Arguments:
							* int control_index: the index of the list control.
							* bool letters: can the user navigate with letters?
							* bool numbers: can the user navigate with the numbers.
							* bool nav_translate = true: should the letters work with the translated alphabet in use?

						Returns:
							bool: true if the settings were successfully set, false otherwise.


						set_list_position:
							Set the user's cursor in a list control.

							`bool audio_form::set_list_position(int control_index, int position = -1, bool speak_new_item = false);`

						Arguments:
							* int control_index: the index of the list.
							* int position = -1: the new cursor position (-1 for no selection).
							* bool speak_new_item = false: should the user be notified of the selection change via speech?

						Returns:
							bool: true if the cursor position was successfully set, false otherwise.


						set_list_properties:
							Sets the properties of a list control.

							`bool audio_form::set_list_propertyes(int control_index, bool multiselect=false, bool repeat_boundary_items=false);`

						Arguments:
							* int control_index: the index of the list.
							* bool multiselect = false: can the user select multiple items in this list?
							* bool repeat_boundary_items = false: do items repeat if you press the arrow keys at the edge of the list?

						Returns:
							bool: true on success, false on failure.


						set_progress:
							Sets the progress percentage on the progress control.

							`bool audio_form::set_progress(int control_index, int value);`

						Arguments:
							* int control_index: the index of the control.
							* int value: the percentage to set.

						Returns:
							bool: true on success, false on failure.

						Remarks:
							This method only works on progress control.


						set_slider:
							Sets the value of the slider control.

							`bool audio_form::set_slider(int control_index, double value, double min = -1, double max = -1);`

						Arguments:
							* int control_index: the index of the control.
							* double value: the value to set.
							* double min = -1: the minimum value to set. This can be omitted.
							* double max = -1: the maximum value to set. This can be omitted.

						Returns:
							bool: true on success, false on failure.

						Remarks:
							This method only works on slider control.


						set_state:
							Set the enabled/visible state of a control.

							`bool audio_form::set_state(int control_index, bool enabled, bool visible);`

						Arguments:
							* int control_index: the index of the control.
							* bool enabled: is the control enabled (e.g. if it's a button, being disabled would make the button unpressable).
							* bool visible: can the user access the control with the navigation commands?

						Returns:
							bool: true if the state was successfully set, false otherwise.


						set_subform:

							Sets a sub form to use on the current form.

							bool audio_form::set_subform(int control_index, audio_form@ f);

						Arguments:

							* int control_index: the control index of the sub form created with `audio_form::create_subform` method.

							* audio_form@ f: an object pointing to an already created form.

						Returns:

							bool: true if the operation was successful, false otherwise. To get error information, look at `audio_form::get_last_error();`.

						Example:

							```NVGT
							#include "form.nvgt"
							// some imaginary global application variables.
							bool logostart = true, menuwrap = false, firespace = false, radar = true;
							// First, lets make a class which stores a category name and the form that the category is linked to.
							class settings_category {
								string name;
								audio_form@ f;
								settings_category(string n, audio_form@ f) {
									this.name = n;
									@this.f = @f;
								}
							}
							void settings_dialog() {
								// Define some categories and settings on each category like this:
								audio_form fc_general;
								fc_general.create_window();
								int f_logostart = fc_general.create_checkbox("Play &logo on startup", logostart);
								int f_menuwrap = fc_general.create_checkbox("Wrap &menus", menuwrap);
								audio_form fc_gameplay;
								fc_gameplay.create_window();
								int f_firespace = fc_gameplay.create_checkbox("Use space instead of control to &fire", firespace);
								int f_radar = fc_gameplay.create_checkbox("Enable the &radar", firespace);
								// Add as many categories as you want this way.
								audio_form f; // The overarching main form.
								f.create_window("Settings", false, true);
								int f_category_list = f.create_tab_panel("&Category"); // The user will select categories from here. Note: you can also use create_list.
								int f_category_display = f.create_subform("General settings", @fc_general); // Now by default, the main form embeds the general category form right there.
								int f_ok = f.create_button("&Save settings", true);
								int f_cancel = f.create_button("Cancel", false, true);
								// Now lets create a structured list of categories that can be browsed based on the class above.
								settings_category@[] categories = {
									settings_category("General", @fc_general),
									settings_category("Gameplay", @fc_gameplay)
								};
								// And then add the list of categories to the form's list.
								for (uint i = 0; i < categories.length(); i++) {
									f.add_list_item(f_category_list, categories[i].name);
								}
								// Focus the form's list position on the general category, then set the form's initial focused control to the category list.
								f.set_list_position(f_category_list, 0);
								f.focus(0);
								settings_category@ last_category = @categories[0]; // A handle to the currently selected category so we can detect changes to the selection.
								// Finally this is the loop that does the rest of the magic.
								while (!f.is_pressed(f_cancel)) {
									wait(5);
									f.monitor();
									int pos = f.get_list_position(f_category_list);
									settings_category@ selected_category = null;
									if (pos > -1 and pos < categories.length())
										@selected_category = @categories[pos];
									if (@last_category != @selected_category) {
										last_category.f.subform_control_index = -1; // Later improvements to audio form will make this line be handled internally.
										last_category.f.focus_silently(0); // Make sure that if the category is reselected, it is focused on the first control.
										@last_category = @selected_category;
										f.set_subform(f_category_display, @selected_category.f);
										f.set_caption(f_category_display, selected_category.name + " settings");
									}
									// The following is a special feature I came up with in stw which makes it so that if you are in the category list, keyboard shortcuts from the entire form will work regardless of category.
									if (f.get_current_focus() == f_category_list and key_down(KEY_LALT) or key_down(KEY_RALT)) {
										for (uint i = 0; i < categories.length(); i++) {
											if (categories[i].f.check_shortcuts(true)) {
												f.set_list_position(f_category_list, i);
												@last_category = @categories[i];
												f.set_subform(f_category_display, @last_category.f);
												f.set_caption(f_category_display, last_category.name + " settings");
												f.focus(f_category_display);
												break;
											}
										}
									}
									// Now we can finally check for the save button.
									if (f.is_pressed(f_ok)) {
										logostart = fc_general.is_checked(f_logostart);
										menuwrap = fc_general.is_checked(f_menuwrap);
										firespace = fc_gameplay.is_checked(f_firespace);
										radar = fc_gameplay.is_checked(f_radar);
										return;
									}
								}
							}
							// Lets make this thing run so we can see it work.
							void main() {
								show_window("test");
								settings_dialog();
							}



						set_text:
							Sets the text of the control.

							`bool audio_form::set_text(int control_index, string new_text);`

						Arguments:
							* int control_index: the index of the control.
							* string new_text: the text to set.

						Returns:
							bool: true on success, false on failure.

						Remarks:
							This method only works on input field, slider, and status bar.


					Properties:
						active:
							Determine if the form is currently active.

							`bool audio_form::active;`


			Enums:
				audioform_errorcodes:
					This enum contains any error values that can be returned by the `audio_form::get_last_error();` function.

					* form_error_none: No error.
					* form_error_invalid_index: you provided a control index that doesn't exist.
					* form_error_invalid_control: you are attempting to do something on an invalid control.
					* form_error_invalid_value: You provided an invalid value.
					* form_error_invalid_operation: you tried to perform an invalid operation.
					* form_error_no_window: you haven't created an audio_form window yet.
					* form_error_window_full: the window is at its maximum number of controls
					* form_error_text_too_long: the text provided is too long.
					* form_error_list_empty: indicates that a list control is empty.
					* form_error_list_full: indicates that a list control is full.
					* form_error_invalid_list_index: the list has no item at that index.
					* form_error_control_invisible: the specified control is invisible.
					* form_error_no_controls_visible: no controls are currently visible.


				control_event_type:
					Lists all possible event types that can be raised.

					* event_none: no event.
					* event_focus: a control gained keyboard focus.
					* event_list_cursor: the cursor changed in a list control.
					* event_text_cursor: the cursor changed in an input box.
					* event_button: a button was pressed.
					* event_checkbox: a checkbox's state has changed.
					* event_slider: a slider's value has been changed.


				control_types:
					This is a complete list of all control types available in the audio form, as well as a brief description of what they do.

					* ct_button: a normal, pressable button.
					* ct_input: any form of text box.
					* ct_checkbox: a checkable/uncheckable control.
					* ct_progress: a progress bar that can both beep and speak.
					* ct_status_bar: a small informational area for your users to get information quickly.
					* ct_list: a list of items.
					* ct_slider: a slider, adjustable to a given percent with the arrow keys.
					* ct_form: a child form.
					* ct_keyboard_area: an area that captures all keys from the user (minus a few critical ones like tab and shift+tab) and lets you handle them, useful for embedding your game controls within your UI.


				text_edit_mode_constants:
					This is a list of constants that specify text editing modes, used mainly with `audio_form::edit_text();`.

					* edit_mode_replace: replace text.
					* edit_mode_trim_to_length: trim the final text to a given length.
					* edit_mode_append_to_end: append text to the end.

					See the `audio_form::edit_text();` documentation for more information.


				text_entry_speech_flags:
					This enum provides constants to be used with the character echo functionality in input boxes.

					* textflag_none: no echo.
					* textflag_characters: echo characters only.
					* textflag_words: echo words only.
					* textflag_characters_words: echo both characters and words.


			Global Properties:
				audioform_input_disable_ralt:
					Set whether or not the right Alt key should be disabled in input boxes, mostly useful for users with non-english keyboards.

					`bool audioform_input_disable_ralt;`


				audioform_keyboard_echo:
					Set the default echo mode for all the forms. Default is `textflag_characters`

					`int audioform_keyboard_echo;`


				audioform_word_separators:
					A list of characters that should be considered word boundaries for navigation in an input box.

					`string audioform_word_separators;`


		Basename Clearing (clear_compiled_basename.nvgt):
			clear_compiled_basename.nvgt:
				This is to solve an edge case you probably won't have to deal with. If you want to have a config.nvgt file for your project for example that defines the compiled_basename pragma, but then you include such a config script in a utility program for your project, there is no way to override the pragma from within that utility script which is not your main project application. Unfortunately pragmas in a file are evaluated before the includes, so you can't override this in a utility programs' main code file.


		BGT Compatibility Layer (bgt_compat.nvgt):
			BGT Compatibility Layer:
				This is an include file that attempts to make old BGT games run in NVGT with minimal modifications, mainly by remapping function names and writing a few convenience wrappers. All function signatures are exactly how they would've been in BGT unless otherwise noted.

				Below, you'll find a list of everything this include does, as well as a reference for what each item roughly maps to in native nVGT.

			Key constants:
				* KEY_PRIOR: KEY_PAGEUP.
				* KEY_NEXT: KEY_PAGEDOWN.
				* KEY_LCONTROL: KEY_LCTRL.
				* KEY_RCONTROL: KEY_RCTRL.
				* KEY_LWIN: KEY_LGUI.
				* KEY_RWIN: KEY_RGUI.
				* KEY_LMENU: KEY_LALT.
				* KEY_RMENU: KEY_RALT.
				* KEY_LBRACKET: KEY_LEFTBRACKET.
				* KEY_RBRACKET: KEY_RIGHTBRACKET.
				* KEY_NUMPADENTER: KEY_NUMPAD_ENTER.
				* KEY_DASH: KEY_MINUS.

			Math functions:
				* absolute: abs.
				* cosine: cos.
				* sine: sin.
				* tangent: tan.
				* arc_cosine: acos.
				* arc_sine: asin.
				* arc_tangent: atan.
				* power: pow.
				* square_root: sqrt.
				* ceiling: ceil.

			Screen reader speech:
				Note that these functions don't map cleanly to how NVGT's API works. As such, substitutions will not be provided here. It is recommended that you use NVGT's built-in screen reader speech functions if you can, they're much more efficient, and much more powerful.

			Constants:
				* JAWS.
				* WINDOW_EYES.
				* SYSTEM_ACCESS.
				* NVDA.

			Functions:
				* screen_reader_is_running.
				* screen_reader_speak.
				* screen_reader_speak_interrupt.
				* screen_reader_stop_speech.

			String functions:
				* string_len: string.length.
				* string_replace: string.replace.
				* string_left: string.substr.
				* string_right: string.slice.
				* string_trim_left: string.substr.
				* string_trim_right: string.slice.
				* string_mid: string.substr.
				* string_is_lower_case: string.is_lower.
				* string_is_upper_case: string.is_upper.
				* string_is_alphabetic: string.is_alphabetic.
				* string_is_digits: string.is_digits.
				* string_is_alphanumeric: string.is_alphanumeric.
				* string_reverse: string.reverse.
				* string_to_lower_case: string.lower.
				* string_to_upper_case: string.upper.
				* string_split: string.split.
				* string_contains: string.find.
				* get_last_error_text: Superseded by exceptions.
				* string_to_number: parse_float.
				* string_compress: string_deflate.
				* string_decompress: string_inflate.

			String encryption/decryption:
				Note that these functions don't work with existing BGT data.
				* string_encrypt.
				* string_decrypt.

			UI functions:
				* show_game_window: show_window.
				* is_game_window_active: is_window_active.


		Character Controller Subsystem (basic_character_controller.nvgt):
			Basic character controller include:

				An abstract character controller that implements a complete framework for 3D character movement without relying on a full physics engine. It accepts input flags for moving forward, backward, strafing, turning, jumping, and toggling crouch, and maintains state flags for conditions such as being grounded, crouching, and running. Internally, the class manages position, velocity, and orientation vectors along with previous states for these quantities.

				It contains methods that update rotation by computing target and current yaw angles, adjusting the forward and right directional vectors accordingly. Vertical movement is handled by applying gravitational acceleration and jump impulses when grounded. Horizontal movement combines input-based directional movement with acceleration, deceleration, and different speed settings for walking, running, and crouching. The class also applies friction--varying between ground and air--and computes step cycles based on distance traveled.

				Additional features include methods for rotating the character by specific degrees, turning around, and snapping to preset angular increments, as well as helper functions that translate yaw values into directional text. Callbacks are provided for position, rotation, vertical and horizontal movement updates, and friction application.

			Usage:

				This class is an abstract base class (ABC). As such, it cannot be directly instantiated. To use it, create a class deriving from `basic_character_controller`. Be sure to call the superclasses constructor within the subclasses constructor, as the superclasses constructor initializes core state and computes general attributes to get you started.

				In your subclass, you are strongly encouraged to override at least one of the callbacks enumerated below. The callbacks notify your application about state changes from within the class and allow you to intercept these changes. For all callbacks returning `bool`, excepting `should_run`, returning `false` will cause the operations performed before that callback is called to be undone. However, other operations performed for each frame will be unaffected unless explicitly specified.

				Do not call any base class callback from within a callback. The behavior is unspecified if this occurs.

				When instantiating your derived class, set any of the below input signals to cause the character controller to take an action. Clear the input signal to cause the action to be halted in the next frame.

				You will also need to compute delta time. Delta time is the duration between frames.

			A note on handedness:

				Handedness defines the orientation convention for three-dimensional coordinate systems. In a left-handed system, the forward direction is defined as the positive Z-axis and the right direction is computed by taking the cross product of the upward vector with the forward vector. In a right-handed system, the forward direction is defined as the negative Z-axis and the right direction is computed by reversing the order of the cross product used in a left-handed system. The controller defaults to a right-handed coordinate system.

				The handedness can be controlled by using the `coordinate_handedness` enumeration. When inheriting from this class, implementors can either control the handedness at instantiation time by passing the mode to the superclasses constructor, or they can alter it on the fly by setting the `handedness` property to the desired mode.

			Input signals:

				The basic character controller is independent of any specific input system. This allows it to be used by both human and machine operators; for example, the player could have a character controller class, as could an AI character in your game.

				To facilitate movement and other operations, you set any of the following input signals. As long as they are true, the appropriate action will be called per frame.

				| **Input Signal** | **Description** |
				| ---------------- | --------------- |
				| `move_forward`   | Causes the character to accelerate in the direction it is facing. |
				| `move_backward`  | Causes the character to accelerate in the opposite direction to which it is facing. |
				| `turn_left`      | Causes the character to rotate counterclockwise around its vertical axis. |
				| `turn_right`     | Causes the character to rotate clockwise around its vertical axis. |
				| `strafe_left`    | Causes the character to move sideways to the left relative to its facing direction. |
				| `strafe_right`   | Causes the character to move sideways to the right relative to its facing direction. |
				| `jump`           | Applies an upward force that moves the character vertically. |

			Callbacks:

				This class implements several callbacks. Override them to change the behavior of the character.

				| **Callback**            | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
				|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
				| `on_position_changed`     | Called after updating the character’s position. By default, this callback verifies the vertical coordinate and, if it is below zero, clamps the vertical position to zero, resets the vertical velocity to zero, and sets the grounded flag to true. If the callback returns false, the entire position update is reverted to the prior state.                                                       |
				| `on_rotation_changed`     | Called after the rotation values (yaw, forward, and right vectors) have been updated. The default implementation does nothing and returns true. If a subclass returns false from this callback, the rotation update is undone and the previous orientation is restored.                                                                                                                                                                                                                                                                      |
				| `on_vertical_movement`    | Called following any modification to the vertical component of the velocity, such as when applying gravity or jump impulses. The default behavior is to return true and to do nothing. Returning false will revert the vertical component back to its previous value.                                                                                                                           |
				| `on_horizontal_movement`  | Called after adjustments are made to the horizontal components of the velocity based on movement inputs. By default, the function returns true and does nothing. If it returns false, the horizontal movement update is restored to the previous horizontal velocity state. |
				| `on_friction_applied`      | Called after friction is applied to the current velocity to dampen motion. The default implementation returns true and does nothing. If a subclass returns false, the velocity changes resulting from friction are undone.                                                                                                                               |
				| `on_step_cycle`           | Called when the cumulative horizontal distance traveled reaches a specified step threshold, typically used to signal step cycle events (such as footstep sounds). The default implementation is a no-op. Undoing this operation is not supported.                                                                                                                                                                                                                             |
				| `should_run`              | Evaluated to determine if the character should be in a running state based on current movement and input flags. The default behavior returns false, meaning the character does not transition into a running state. This callback is purely evaluative, does not modify state, and its return value affects only the choice of movement speed during horizontal motion calculations. |

			Fields:

				When implementing callbacks or the constructor of a subclass, many fields are made available to grant you access to the internal state of the character. These fields are as follows:

				| **Field**           | **Description**                                                                                                                                           |
				|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
				| `is_grounded`         | Boolean indicating whether the character is in contact with the ground.                                                                                 |
				| `is_crouching`        | Boolean indicating whether the character is in a crouched state.                                                                                          |
				| `is_running`          | Boolean indicating whether the character is in a running state.                                                                                           |
				| `position`            | Vector representing the current position of the character in the game world.                                                                              |
				| `velocity`            | Vector representing the current rate of change of the character's position.                                                                               |
				| `forward`             | Vector representing the current forward direction of the character.                                                                                     |
				| `up`                  | Vector representing the upward direction relative to the character, commonly used as a vertical reference.                                                  |
				| `right`               | Vector representing the rightward direction, computed as the normalized cross product of the up vector and the forward vector.                             |
				| `old_position`        | Vector storing the character's position from the previous update cycle.                                                                                   |
				| `old_velocity`        | Vector storing the character's velocity from the previous update cycle.                                                                                   |
				| `old_forward`         | Vector storing the character's forward direction from the previous update cycle.                                                                          |
				| `old_up`              | Vector storing the character's up direction from the previous update cycle.                                                                               |
				| `old_right`           | Vector storing the character's right direction from the previous update cycle.                                                                            |
				| `max_speed`           | Float representing the maximum speed that can be reached by the character.                                                                                |
				| `walk_speed`          | Float defining the target speed when the character is walking.                                                                                            |
				| `run_speed`           | Float defining the target speed when the character is running.                                                                                            |
				| `crouch_speed`        | Float defining the target speed when the character is crouching.                                                                                          |
				| `acceleration`        | Float that defines the rate at which the character's velocity increases when movement input is applied.                                                     |
				| `deceleration`        | Float that defines the rate at which the character's velocity decreases when movement input is not applied.                                                 |
				| `air_control`         | Float multiplier that scales the acceleration when the character is airborne, affecting maneuverability in the air.                                          |
				| `rotation_speed`      | Float that defines the rate of angular change (rotation) of the character's orientation, measured in radians per second.                |
				| `current_speed`       | Float representing the current magnitude of the character's horizontal movement speed.                                                                     |
				| `jump_force`          | Float representing the magnitude of the upward impulse applied when the character jumps.                                                                   |
				| `gravity`             | Float representing the acceleration due to gravity that affects the character's vertical velocity, measured in meters per second squared (`m/s^2`).                                                        |
				| `ground_friction`     | Float representing the frictional force applied to the character while it is in contact with the ground, reducing its horizontal velocity.                 |
				| `air_friction`        | Float representing the frictional force applied to the character while airborne, reducing its horizontal velocity.                                          |
				| `player_height`       | Float representing the current height of the character, which can be modified by actions like crouching. Currently unused internally.                                                    |
				| `crouch_height`       | Float defining the character's height when in a crouched state. Currently unused internally.                                                                                            |
				| `stand_height`        | Float defining the character's height when standing upright. Currently unused internally.                                                                                              |
				| `mass`                | Float representing the character's mass, in kilograms. Currently unused internally.                                        |
				| `step_distance`       | Float used as an incremental measure for calculating the distance required to complete one step cycle.                                                     |
				| `distance_traveled`   | Float accumulating the horizontal distance the character has moved since the last reset of the step cycle.                                                   |
				| `step_cycle`          | Float representing the progress through the current step cycle, often normalized to a value between 0 and 1.                                                  |
				| `run_step_length`     | Float defining the distance threshold for a single step cycle when the character is running.                                                               |
				| `walk_step_length`    | Float defining the distance threshold for a single step cycle when the character is walking.                                                               |
				| `crouch_step_length`  | Float defining the distance threshold for a single step cycle when the character is crouching.                                                             |
				| `yaw`                 | Float representing the current rotational angle (yaw) of the character around the vertical axis, in degrees. You should normally not neeed to change this field; doing so will cause an (unnatural) "snapping" effect, which you may not typically want.                                                           |
				| `target_yaw`          | Float representing the desired yaw toward which the character is currently rotating, in degrees. To naturally rotate the character to a given yaw, set this field and allow the character to reach that yaw over time.                                                                |
				| `old_yaw`             | Float storing the yaw angle from the previous update cycle.                                                                                              |


			Classes:
				basic_character_controller:
					Methods:
						`realign_to_nearest_degree`:

							Instructs the controller that it should reorient itself such that it is facing the closest valid compass point to it's current yaw.

							```nvgt
							void realign_to_nearest_degree(const float step=2.8125) final;

						Parameters:

							* `const float step`: Used for snapping degrees to the nearest compass point, this parameter allows you to change the granularity of snapping behaviors.



						`rotate_left_by`:

							Instructs the controller that it should begin rotation of the character to the left by the given amount on it's vertical axis.

							```nvgt
							void rotate_left_by(const float degree, const float step = 2.8125, const bool snap = false) final;

						Parameters:

							* `const float degree`: the amount by which to rotate the character, in degrees.
							* `const float step`: Used for snapping degrees to the nearest compass point, this parameter allows you to change the granularity of snapping behaviors.
							* `const bool snap`: Determines whether the target yaw will be snapped to the nearest valid compass point, or whether it will be used as is.

						Remarks:

							The methods `rotate_left_by` and `rotate_right_by` update the character's yaw by subtracting or adding a specified angle, respectively. When the `snap` parameter is enabled, the input angle is adjusted, or "snapped", to the nearest multiple defined by the `step` parameter. By default, the system uses a `step` value of `2.8125` degrees, which results from dividing 360 degrees by 128, corresponding to a Mariner’s compass with 128 discrete points. For cases where a different level of angular granularity is preferred, the `step` value can be recalculated; for instance, using `360/4` (`90.0` degrees) restricts rotation to the four cardinal directions, or using `360/8` (`45.0` degrees) allows both cardinal and intercardinal directions. Once the angle is snapped (if applicable), the method updates the target yaw, which is used in subsequent rotation updates to interpolate the actual yaw toward the target value.

						`rotate_right_by`:

							Instructs the controller that it should begin rotation of the character to the right by the given amount on it's vertical axis.

							```nvgt
							void rotate_right_by(const float degree, const float step = 2.8125, const bool snap = false) final;

						Parameters:

							* `const float degree`: the amount by which to rotate the character, in degrees.
							* `const float step`: Used for snapping degrees to the nearest compass point, this parameter allows you to change the granularity of snapping behaviors.
							* `const bool snap`: Determines whether the target yaw will be snapped to the nearest valid compass point, or whether it will be used as is.

						Remarks:

							The methods `rotate_left_by` and `rotate_right_by` update the character's yaw by subtracting or adding a specified angle, respectively. When the `snap` parameter is enabled, the input angle is adjusted, or "snapped", to the nearest multiple defined by the `step` parameter. By default, the system uses a `step` value of `2.8125` degrees, which results from dividing 360 degrees by 128, corresponding to a Mariner’s compass with 128 discrete points. For cases where a different level of angular granularity is preferred, the `step` value can be recalculated; for instance, using `360/4` (`90.0` degrees) restricts rotation to the four cardinal directions, or using `360/8` (`45.0` degrees) allows both cardinal and intercardinal directions. Once the angle is snapped (if applicable), the method updates the target yaw, which is used in subsequent rotation updates to interpolate the actual yaw toward the target value.

						`toggle_crouch`:

							Toggles whether the character is in a crouching state.

							```nvgt
							void toggle_crouch() final;



						`turn_around`:

							Instructs the controller that the character should perform a full 180-degree rotation to the right.

							```nvgt
							void turn_around() final;

						Remarks:

							This function is most useful when the character is facing a cardinal direction.

						`update`:

							Instructs the controller that a new frame is about to begin and it should perform all necessary actions to realize field/property changes.

							```nvgt
							void update(const float delta_time) final;

						Parameters:

							* `const float delta_time`: the difference between the time of the previous frame and this new frame. This value should be reasonably consistent; drastic changes will cause dramatic changes in the controller's internal state.

						Remarks:

							This function should be called once per iteration of your game loop. Your loop should take as little time as possible, and there should be as little time between `update` calls as you can manage. This function should be called *after* making any changes to public properties or fields; calling it *before* changes are made will work, but changes will not be realized until the subsequent frame.

							Warning: do not insert extreme delays between calls to this function. If the delta time is too extreme, the simulation and physics will be unpredictable.


					Properties:
						`grounded`:

							Determines if the player is on the ground for simulation purposes.

							```nvgt
							void set_grounded(const bool grounded) property final;

						Remarks:

							What is defined as the "ground" is left up to subclasses. For sane behavior, the "ground" is, by default, a vertical axis of 0 in the position vector. However, you are free to change this by overriding the `on_position_changed` callback. To lock the player at whatever you determine is the ground:

							1. Set the vertical axis with respect to position and velocity to zero. Do not modify any other axis of either vector.
							2. Set this property to true.

							Note that gravity will always be applied, regardless of whether the player is on the ground or airborn. When overriding the `on_position_changed` callback, always implement some form of ground detection; should you fail to do so, the character will fall forever.

						`handedness`:

							Determines whether this controller uses a left-or right-handed coordinate system. The default is to use a right-handed coordinate system.

							```nvgt
							coordinate_handedness get_handedness() property final;
							void set_handedness(const coordinate_handedness value) property final;

						Remarks:

							For information on handedness and exactly how this alters the behavior of the controller, see the introduction to this class.

							When changing this property, the forward and right vectors are automatically recomputed. No other properties are modified.

							It is strongly recommended that the handedness of the controller only be altered when operating in a setting where a left-handed coordinate system is required (e.g., for operating with an audio engine which defaults to it). Changing this property should be rare in practice.

							It is safe to assume that the value of this property is always in a valid state.

						Exceptions:

							An assertion failure occurs if neither of the valid enumeration values is provided to the setter of this property. An assertion failure also occurs in the constructor if an invalid value is supplied.

						`target_yaw`:

							Determines the target yaw to which the character should rotate.

							```nvgt
							void set_target_yaw(const float value) property;
							float get_target_yaw() property;

						Remarks:

							This property can be used to instruct the controller to orient the player to any arbitrary yaw. If the value exceeds 360 or is negative, the behavior is undefined. The controller will attempt to automatically correct for this problem when handling rotation, but depending on this behavior is strongly discouraged.

							Note that setting this property does not immediately cause a rotation to occur. Rotation will be performed per frame (that is, every time `update` is called with a steady `delta_time`). This ensures that rotation feels natural to the player. If you wish to cause an immediate rotation, expose a way for your character controller subclass to force `yaw` to a specific value. This behavior is, however, discouraged, but it should not break the simulation if you do so.

						`yaw`:

							Determines the current yaw of the character.

							```nvgt
							float get_yaw() property;
							private void set_yaw(const float value) property;

						Remarks:

							Setting this property is explicitly disallowed to subclasses because it is typically a very unusual thing to do. Although it does not break the simulation, it is not normally something implementors should contemplate. There are alternative methods of achieving this, such as setting the rotation speed to a very high value. If you do wish to be able to set this property, you are free to override this function.

			Enums:
				`coordinate_handedness`:

					Determines the handedness of the coordinate system used by this controller.

					| Constant | Description |
					| --- | --- |
					| `COORDINATE_HANDEDNESS_LEFT` | Instructs this controller to use a left-handed coordinate system. |
					| `COORDINATE_HANDEDNESS_RIGHT` | Instructs this controller to use a right-handed coordinate system. |


			Functions:
				`nearest_compass_point`:

					Snaps the provided yaw to the closest compass point based on `step`.

					```nvgt
					float nearest_compass_point(const float deg, const float step = 2.8125);

				Parameters:

					* `const float deg`: the yaw that should be adjusted.
					* `const float step`: the granularity of the adjustment performed.


				`snap_to_degree`:

					Attempts to snap the degree to any of the given points. The behavior is undefined if no points are provided.

					```nvgt
					float snap_to_degree(const float deg, const float[] snaps);

				Parameters:

					* `const float deg`: the current yaw
					* `const float[] snaps`: the list of valid compass points to pick from.

				Remarks:

					The points that callers provide should be whole numbers only. Fractional numbers may work, but are not guaranteed.

				Exceptions:

					An assertion error is triggered if the provided yaw is less than 0 or greater than 360.

				`translate_yaw_to_direction`:

					Translates the given yaw to a direction string.

					```nvgt
					string translate_yaw_to_direction(const float yaw);

				Parameters:

					* `const float yaw`: the yaw to translate to a direction.

				Returns:

					`string`: the direction string that this yaw would correspond to.

				Remarks:

					This function uses the Mariner's compass, which has 128 points and has some direction strings not typically scene outside specific applications or domains. This function includes these rarer points for completeness.

					This function is a utility function provided so that it is one less thing you need to implement.

		Character Rotation (rotation.nvgt):
			character rotation:
				This include contains functions for moving a rotating character in a 2D or 3D game.

			Disclaimer:
				Though these have been improved over the years and though I do use this myself for Survive the Wild and my other games, it should be understood that I started writing this file in BGT when I was only 12 or 13 years old and it has only been getting improved as needed. The result is that anything from the math to the coding decisions may be less than perfect, putting it kindly. You have been warned!


			Functions:
				calculate_theta:

					Calculate the radians value for a given angle in degrees.

					double calculate_theta(double deg);

				Arguments:

					* double deg: the angle to convert, in degrees.

				Returns:

					double: the specified angle in radians.

				Example:

					```NVGT
					#include "rotation.nvgt"
					void main() {
						alert("Info", "45 degrees in radians is " + calculate_theta(45));
					}



				get_1d_distance:

					Get the distance between two points on the x axis.

					double get_1d_distance(double x1, double x2);

				Arguments:

					* double x1: the first point.

					* double x2: the second point.

				Returns:

					double: the distance between the two points on the x axis.

				Example:

					```NVGT
					#include "rotation.nvgt"
					void main() {
						double x1 = 2.0;
						double x2 = 6.8;
						alert("The distance between " + x1 + " and " + x2 + " is", get_1d_distance(x1, x2));
					}



				get_2d_distance:

					Get the distance between two x/y points.

					double get_2d_distance(double x1, double y1, double x2, double y2);

				Arguments:

					* double x1: the first x point.

					* double y1: the first y point.

					* double x2: the second x point.

					* double y2: the second y point.

				Returns:

					double: the distance between the two points.

				Remarks:

					This function uses the Euclidean distance formula, meaning it will return the possible closest distance between the two points.

				Example:

					```NVGT
					#include "rotation.nvgt"
					void main() {
						double x1 = 2.0, y1 = 0.3;
						double x2 = 6.8, y2 = 9.45;
						alert("The distance between (" + x1 + ", " + y1 + ") and (" + x2 + ", " + y2 + ") is", get_2d_distance(x1, y1, x2, y2));
					}



				get_3d_distance:

					Get the distance between two x/y/z points.

					double get_3d_distance(double x1, double y1, double z1, double x2, double y2, double z2);

				Arguments:

					* double x1: the first x point.

					* double y1: the first y point.

					* double z1: the first z point.

					* double x2: the second x point.

					* double y2: the second y point.

					* double z2: the second z point.

				Returns:

					double: the distance between the two points.

				Remarks:

					This function uses the Euclidean distance formula, meaning it will return the possible closest distance between the two points.

				Example:

					```NVGT
					#include "rotation.nvgt"
					void main() {
						double x1 = 2.0, y1 = 0.3, z1 = 0.0;
						double x2 = 6.8, y2 = 9.45, z2 = 1.942;
						alert("The distance between (" + x1 + ", " + y1 + ", " + z1 + ") and (" + x2 + ", " + y2 + ", " + z2 + ") is", get_2d_distance(x1, y1, x2, y2));
					}



			Global Properties:
				direction constants:
					This is a list of the various direction constants present in the rotation include. Each constant will be listed, as well as what it represents.

					* int north: 0 degrees.
					* int northeast: 45 degrees.
					* int east: 90 degrees.
					* int southeast: 135 degrees.
					* int south: 180 degrees.
					* int southeast: 225 degrees.
					* int west: 270 degrees.
					* int northwest: 315 degrees.
					* int half_up: 45 degrees (upwards).
					* int straight_up: 90 degrees (upwards).
					* int half_down: 135 degrees (downwards).
					* int straight_down: 180 degrees (downwards).


				pi:

					Holds 32 digits of PI.

					const double pi;

				Example:

					```NVGT
					#include "rotation.nvgt"
					void main() {
						alert("32 digits of PI is", pi);
					}



		Dictionary Retrieval (dget.nvgt):
			Dictionary retrieval functions:
				The way you get values out of AngelScript dictionaries by default is fairly annoying, mainly due to its usage of out values instead of returning them. Hence this include, which attempts to simplify things.


			functions:
				dgetb:
					Get a boolean value out of a dictionary.

					`bool dgetn(dictionary@ the_dictionary, string key, bool def = false);`

				Arguments:
					* dictionary@ the_dictionary: a handle to the dictionary to get the value from.
					* string key: the key of the value to look up.
					* bool def = false: the value to return if the key wasn't found.

				Returns:
					bool: the value for the particular key in the dictionary, or the default value if not found.


				dgetn:
					Get a numeric value out of a dictionary.

					`double dgetn(dictionary@ the_dictionary, string key, double def = 0.0);`

				Arguments:
					* dictionary@ the_dictionary: a handle to the dictionary to get the value from.
					* string key: the key of the value to look up.
					* double def = 0.0: the value to return if the key wasn't found.

				Returns:
					double: the value for the particular key in the dictionary, or the default value if not found.


				dgets:
					Get a string value out of a dictionary.

					`string dgets(dictionary@ the_dictionary, string key, string def = 0.0);`

				Arguments:
					* dictionary@ the_dictionary: a handle to the dictionary to get the value from.
					* string key: the key of the value to look up.
					* string def = "": the value to return if the key wasn't found.

				Returns:
					string: the value for the particular key in the dictionary, or the default value if not found.


				dgetsl:
					Get a string array out of a dictionary.

					`string[] dgetsl(dictionary@ the_dictionary, string key, string[] def = []);`

				Arguments:
					* dictionary@ the_dictionary: a handle to the dictionary to get the value from.
					* string key: the key of the value to look up.
					* string[] def = []: the value to return if the key wasn't found.

				Returns:
					string[]: the value for the particular key in the dictionary, or the default value if not found.

				Remarks:
					The default value for this function is a completely empty (but initialized) string array.


		INI Reader and Writer (ini.nvgt):
			INI Reader and Writer:
				This is a class designed to read and write INI configuration files.

				I can't promise that the specification will end up being followed to the letter, but I'll try. At this time, though the order of keys and sections will remain the same, the whitespace, comment, and line structure of an externally created ini file may not remain in tact if that file is updated and rewritten using this include.


			classes:
				ini:
					This constructor just takes an optional filename so you can load an INI file directly on object creation if you want. Note though that doing it this way makes it more difficult to instantly tell if there was a problem do to the lack of a return value, so you must then evaluate ini::get_error_line() == 0 to verify a successful load.

					`ini(string filename = "");`

				Arguments:
					* string filename = "": an optional filename to load on object creation.


					methods:
						clear_section:
							Deletes all keys from the given section without deleting the section itself.

							`bool ini::clear_section(string section);`

						Arguments:
							* string section: the name of the section to clear.

						Returns:
							bool: true if the section was successfully cleared, false if it doesn't exist.


						create_section:
							Creates a new INI section with the given name.

							`bool ini::create_section(string section_name);`

						Arguments:
							* string section_name: the name of the section to create.

						Returns:
							bool: true if the section was successfully created, false otherwise.


						delete_key:
							Delete a key from a given section.

							`bool ini::delete_key(string section, string key);`

						Arguments:
							* string section: the name of the section the key is stored in (if any).
							* string key: the name of the key to delete.

						Returns:
							bool: true if the key was successfully deleted, false and sets an error if the key you want to delete doesn't exist or if the key name is invalid.


						delete_section:
							Delete the given section.

							`bool ini::delete_section(string section);`

						Arguments:
							* string section: the name of the section to delete (set this argument to a blank string to delete all sectionless keys).

						Returns:
							bool: true if the section was successfully deleted, false otherwise.


						dump:
							Dump all loaded data into a string, such as what's used by the save function, or so that you can encrypt it, pack it or such things.

							`string ini::dump(bool indent = false);`

						Arguments:
							* bool indent = false:  If this is set to true, all keys in every named section will be proceeded with a tab character in the final output.

						Returns:
							string: the entire INI data as a string.


						get_bool:
							Fetch a boolean value from the INI data given a section and key.

							`bool ini::get_bool(string section, string key, bool default_value = false);`

						Arguments:
							* string section: the section to get the value from (if any).
							* string key: the key of the value.
							* bool default_value = false: the default value to return if the key isn't found.

						Returns:
							bool: the value at the particular key if found, the default value if not.

						Remarks:
							All getters will use this format, and if one returns a default value (blank string, an int that equals 0, a boolean that equals false etc), and if you want to know whether the key actually existed, use the error checking system.


						get_double:
							Fetch a double from the INI data given a section and key.

							`double ini::get_double(string section, string key, double default_value = 0.0);`

						Arguments:
							* string section: the section to get the value from (if any).
							* string key: the key of the value.
							* double default_value = 0.0: the default value to return if the key isn't found.

						Returns:
							double: the value at the particular key if found, the default value if not.

						Remarks:
							All getters will use this format, and if one returns a default value (blank string, an int that equals 0, a boolean that equals false etc), and if you want to know whether the key actually existed, use the error checking system.


						get_error_line:
							Return the line the last error took place on if applicable. This does not clear the error information, since one may wish to get the line number and the text which are in 2 different functions. So make sure to call this function before `ini::get_error_text()` if the line number is something you're interested in retrieving.

							`int ini::get_error_line();`

						Returns:
							int: the line number of the last error, if any. A return value of -1 means that this error is not associated with a line number, and 0 means there is no error in the first place.


						get_error_text:
							Returns the last error message, almost always used if an INI file fails to load and you want to know why. This function also clears the error, so to figure out the line, call `ini::get_error_line()` before calling this.

							`string ini::get_error_text();`

						Returns:
							string: the last error message, if any.


						get_string:
							Fetch a string from the INI data given a section and key.

							`string ini::get_string(string section, string key, string default_value = "");`

						Arguments:
							* string section: the section to get the value from (if any).
							* string key: the key of the value.
							* string default_value = "": the default value to return if the key isn't found.

						Returns:
							string: the value at the particular key if found, the default value if not.

						Remarks:
							All getters will use this format, and if one returns a default value (blank string, an int that equals 0, a boolean that equals false etc), and if you want to know whether the key actually existed, use the error checking system.


						is_empty:
							Determine if the INI object has no data in it.

							`bool ini::is_empty();`

						Returns:
							bool: true if there is no data loaded into this ini object, false otherwise.


						key_exists:
							Determine if a particular key exists in the INI data.

							`bool ini::key_exists(string section, string key);`

						Arguments:
							* string section: the name of the section to look for the key in.
							* string key: the name of the key.

						Returns:
							bool: true if the specified key exists, false otherwise.

						Remarks:
							An error will be accessible from the error system if the given section doesn't exist.


						list_keys:
							List all key names in a given section.

							`string[]@ ini::list_keys(string section);`

						Arguments:
							* string section: the section to list keys from(pass a blank string for all sectionless keys as usual).

						Returns:
							string[]@: a handle to an array containing all the keys.  An empty array means that the section is either blank or doesn't exist, the latter being able to be checked with the error system.


						list_sections:
							List all section names that exist.

							`string[]@ list_sections(bool include_blank_section = false);`

						Arguments:
							* bool include_blank_section = false: Set this argument to true if you wish to include the empty element at the beginning of the list for the keys that aren't in sections, for example for automatic data collection so you don't have to insert yourself when looping through.

						Returns:
							string[]@: a handle to an array containing all the key names.


						list_wildcard_sections:
							Returns all section names containing a wildcard identifier. This way if searching through a file containing many normal sections and a few wildcard sections, it is possible to query only the wildcards for faster lookup.

							`string[]@ ini::list_wildcard_sections();`

						Returns:
							string[]@: a handle to an array containing all the wildcard sections.


						load:
							Load an INI file.

							`bool ini::load(string filename, bool robust = true);`

						Arguments:
							* string filename: the name of the ini file to load.
							* bool robust = true: if true, a temporary backup copy of the ini data will be created before saving, and it'll be restored on error. This is slower and should only be used when necessary, but insures 0 data loss.

						Returns:
							bool: true if the ini data was successfully loaded, false otherwise.


						load_string:
							This function loads INI data stored as a string, doing it this way insures that ini data can come from any source, such as an encrypted string if need be.

							`bool ini::load_string(string data, string filename = "*");`

						Arguments:
							* string data: the INI data to load (as a string).
							* string filename = "*": the new filename to set on the INI object, if any.

						Returns:
							bool: true if the data was successfully loaded, false otherwise.

						Remarks:
							Input data is expected to have CRLF line endings.


						reset:
							Resets all variables to default. You can call this yourself and it is also called by loading functions to clear data from partially successful loads upon error.

							`void ini::reset(bool make_blank_section = true);

						Arguments:
							* bool make_blank_section = true: this argument is internal, and exists because the `ini::load_string()` function creates that section itself.


						save:
							Save everything currently loaded to a file.

							`bool ini::save(string filename, bool indent = false);`

						Arguments:
							* string filename: the name of the file to write to.
							* bool indent = false:  If this is set to true, all keys in every named section will be proceeded with a tab character in the final output.

						Returns:
							bool: true if the data was successfully saved, false otherwise.


						save_robust:
							This function is similar to `ini::save()`, but it first performs a temporary backup of any existing data, then restores that backup if the saving fails. This is slower and should only be used when necessary, but should insure 0 data loss.

							`bool ini::save_robust(string filename, bool indent = false);`

						Arguments:
							* string filename: the name of the file to write to.
							* bool indent = false:  If this is set to true, all keys in every named section will be proceeded with a tab character in the final output.

						Returns:
							bool: true if the data was successfully saved, false otherwise.


						section_exists:
							Determine if a particular section exists in the INI data.

							bool ini::section_exists(string section);`

						Arguments:
							* string section: the name of the section to check for.

						Returns:
							bool: true if the section exists, false if not.


						set_bool:
							Set a boolean value in the INI data given a section name, a key and a value.

							`bool ini::set_bool(string section, string key, bool value);`

						Arguments:
							* string section: the section to put this key/value pair in (leave blank to add at the top of the file without a section).
							* string key: the name of the key.
							* bool value: the value to set.

						Returns:
							bool: true if the value was successfully written, false otherwise.

						Remarks:
							All of the other setters use this format. If the key exists already, the value, of course, will be updated.


						set_double:
							Set a double in the INI data given a section name, a key and a value.

							`bool ini::set_double(string section, string key, double value);`

						Arguments:
							* string section: the section to put this key/value pair in (leave blank to add at the top of the file without a section).
							* string key: the name of the key.
							* double value: the value to set.

						Returns:
							bool: true if the double was successfully written, false otherwise.

						Remarks:
							All of the other setters use this format. If the key exists already, the value, of course, will be updated.


						set_string:
							Set a string in the INI data given a section name, a key and a value.

							`bool ini::set_string(string section, string key, string value);`

						Arguments:
							* string section: the section to put this key/value pair in (leave blank to add at the top of the file without a section).
							* string key: the name of the key.
							* string value: the value to set.

						Returns:
							bool: true if the string was successfully written, false otherwise.

						Remarks:
							All of the other setters use this format. If the key exists already, the value, of course, will be updated.


					properties:
						loaded_filename:
							Contains the filename of the currently loaded INI data.

							`string loaded_filename;`


		Instance Management (instance.nvgt):
			Instance Management:
				This include provides the `instance` class, which allows you to manage the way multiple instances of your game are handled. For example, you could use this class to check if your game is already running, or stop it from running until only one instance exists.


			Classes:
				instance:
					Methods:
						wait_until_standalone:
							This method will make any instances of your game block until there's only one instance still alive.

							`void instance::wait_until_standalone();`


					Properties:
						is_already_running:

							Determines if an instance of the application is already running.

							bool instance::is_already_running;

						Example:

							```NVGT
							#include "instance.nvgt"
							void main() {
								instance example("instance_checker_example");
								if (example.is_already_running) {
									alert("Info", "The script is already running.");
									exit();
								}
								alert("Info", "After you press OK, you'll have 15 seconds to run this script again to see the result");
								wait(15000);
							}



		Legacy Sound Manager (sound_pool.nvgt):
			Classes:
				sound_pool:
					This class provides a convenient way of managing sounds in an environment, with 1 to 3 dimensions. The sound_pool_item class holds all the information necessary for one single sound in the game world. Note that you should not make instances of the sound_pool_item class directly but always use the methods and properties provided in the sound_pool class.

					`sound_pool(int default_item_size = 100);`

				Arguments:
					* int default_item_size = 100: the number of sound items to be initialized by default.

					Methods:
						destroy_all:
							destroy all sounds.

							`void sound_pool::destroy_all();`


						destroy_sound:
							Destroy a sound.

							`bool sound_pool::destroy_sound(int slot);`

						Arguments:
							* int slot: the slot of the sound you wish to destroy.

						Returns:
							bool: true if the sound is removed successfully, false otherwise.


						pause_all:
							Pauses all sounds.

							`void sound_pool::pause_all();`


						pause_sound:
							Pauses the sound.

							`bool sound_pool::pause_sound(int slot);`

						Arguments:
							* int slot: the sound slot you wish to pause.

						Returns:
							bool: true if the sound was paused, false otherwise.


						play_1d:
							Play a sound in 1 dimension and return a slot.

							`int sound_pool::play_1d(string filename, pack@ packfile, float listener_x, float sound_x, bool looping, bool persistent = false);`

						Arguments:
							* string filename: the file to play.
							* pack@ packfile: the pack to use. This can be omitted.
							* float listener_x: the listener coordinates in X form.
							* float sound_x: the coordinates of the sound in X form.
							* bool looping: should the sound play continuously?
							* bool persistent = false: should the sound be cleaned up once the sound is finished playing?

						Returns:
							int: the index of the sound which can be modified later, or -1 if error. This method may return -2 if the sound is out of earshot.

						Remarks:
							If the looping parameter is set to true and the sound object is inactive, the sound is still considered to be active as this just means that we are currently out of earshot. A non-looping sound that has finished playing is considered to be dead, and will be cleaned up if it is not set to be persistent.


						play_2d:
							Play a sound in 2 dimensions and return a slot.

							1. `int sound_pool::play_2d(string filename, pack@ packfile, float listener_x, float listener_y, float sound_x, float sound_y, bool looping, bool persistent = false);`
							2. `int sound_pool::play_2d(string filename, pack@ packfile, float listener_x, float listener_y, float sound_x, float sound_y, double rotation, bool looping, bool persistent = false);`
							3. `int sound_pool::play_2d(string filename, float listener_x, float listener_y, float sound_x, float sound_y, bool looping, bool persistent = false);`
							4. `int sound_pool::play_2d(string filename, float listener_x, float listener_y, float sound_x, float sound_y, double rotation, bool looping, bool persistent = false);`

						Arguments:
							* string filename: the file to play.
							* pack@ packfile: the pack to use (optional).
							* float listener_x, listener_y: the listener coordinates in X, Y form.
							* float sound_x, sound_y: the coordinates of the sound in X, Y form.
							* double rotation: the listener's rotation (optional).
							* bool looping: should the sound play continuously?
							* bool persistent = false: should the sound be cleaned up once the sound is finished playing?

						Returns:
							int: the index of the sound which can be modified later, or -1 if error. This method may return -2 if the sound is out of earshot.

						Remarks:
							If the looping parameter is set to true and the sound object is inactive, the sound is still considered to be active as this just means that we are currently out of earshot. A non-looping sound that has finished playing is considered to be dead, and will be cleaned up if it is not set to be persistent.


						play_3d:
							Play a sound in 3 dimensions and return a slot.

							1. `int sound_pool::play_3d(string filename, pack@ packfile, float listener_x, float listener_y, float listener_z, float sound_x, float sound_y, float sound_z, double rotation, bool looping, bool persistent = false);`
							2. `int sound_pool::play_3d(string filename, pack@ packfile, vector listener, vector sound_coordinate, double rotation, bool looping, bool persistent = false);`

						Arguments (1):
							* string filename: the file to play.
							* pack@ packfile: the pack to use. This can be omitted
							* float listener_x, listener_y, listener_z: the listener coordinates in X, Y, Z form.
							* float sound_x, sound_y, sound_z: the coordinates of the sound in X, Y, Z form.
							* double rotation: the listener's rotation.
							* bool looping: should the sound play continuously?
							* bool persistent = false: should the sound be cleaned up once the sound is finished playing?

						Arguments (2):
							* string filename: the file to play.
							* pack@ packfile: the pack to use. This can be omitted
							* vector listener: the coordinates of listener in vector form.
							* vector sound_coordinate: the coordinates of the sound in vector form.
							* double rotation: the listener's rotation.
							* bool looping: should the sound play continuously?
							* bool persistent = false: should the sound be cleaned up once the sound is finished playing?

						Returns:
							int: the index of the sound which can be modified later, or -1 if error. This method may return -2 if the sound is out of earshot.

						Remarks:
							If the looping parameter is set to true and the sound object is inactive, the sound is still considered to be active as this just means that we are currently out of earshot. A non-looping sound that has finished playing is considered to be dead, and will be cleaned up if it is not set to be persistent.


						play_extended:
							Play a sound and return a slot. This method has many parameters that can be customized.

							`int sound_pool::play_extended(int dimension, string filename, pack@ packfile, float listener_x, float listener_y, float listener_z, float sound_x, float sound_y, float sound_z, double rotation, int left_range, int right_range, int backward_range, int forward_range, int lower_range, int upper_range, bool looping, double offset, float start_pan, float start_volume, float start_pitch, bool persistent = false, mixer@ mix = null, string[]@ fx = null, bool start_playing = true, double theta = 0);`

						Arguments:
							* int dimension: the number of dimension to play on, 1, 2, 3.
							* string filename: the file to play.
							* pack@ packfile: the pack to use.
							* float listener_x, listener_y, listener_z: the listener coordinates in X, Y, Z form.
							* float sound_x, sound_y, sound_z: the coordinates of the sound in X, Y, Z form.
							* double rotation: the listener's rotation.
							* int left_range, right_range, backward_range, forward_range, lower_range, upper_range: the range of the sound.
							* bool looping: should the sound play continuously?
							* double offset: the number of milliseconds for the sound to start playing at.
							* float start_pan: the pan of the sound. -100 is left, 0 is middle, and 100 is right.
							* float start_volume: the volume of the sound. 0 is maximum and -100 is silent.
							* float start_pitch: the pitch of the sound.
							* bool persistent = false: should the sound be cleaned up once the sound is finished playing?
							* mixer@ mix = null: the mixer to attach to this sound.
							* string[]@ fx = null: array of effects to be set.
							* bool start_playing = true: should the sound play the moment this function is executed?
							* double theta = 0: the theta calculated by `calculate_theta` function in rotation.nvgt include.

						Returns:
							int: the index of the sound which can be modified later, or -1 if error. This method may return -2 if the sound is out of earshot.

						Remarks:
							If the looping parameter is set to true and the sound object is inactive, the sound is still considered to be active as this just means that we are currently out of earshot. A non-looping sound that has finished playing is considered to be dead, and will be cleaned up if it is not set to be persistent.


						play_stationary:
							Play a stationary sound and return a slot.

							1. `int sound_pool::play_stationary(string filename, bool looping, bool persistent = false);`
							2. `int sound_pool::play_stationary(string filename, pack@ packfile, bool looping, bool persistent = false);`

						Arguments (1):
							* string filename: the file to play.
							* bool looping: should the sound play continuously?
							* bool persistent = false: should the sound be cleaned up once the sound is finished playing?

						Arguments (2):
							* string filename: the file to play.
							* pack@ packfile: the pack to use.
							* bool looping: should the sound play continuously?
							* bool persistent = false: should the sound be cleaned up once the sound is finished playing?

						Returns:
							int: the index of the sound which can be modified later, or -1 if error. This method may return -2 if the sound is out of earshot.

						Remarks:
							If the looping parameter is set to true and the sound object is inactive, the sound is still considered to be active as this just means that we are currently out of earshot. A non-looping sound that has finished playing is considered to be dead, and will be cleaned up if it is not set to be persistent.

							This method will play the sound in stationary mode. This means that sounds played by this method have no movement updates as if they are stationary, and coordinate update functions will not work.

						resume_all:
							Resumes all sounds.

							`void sound_pool::resume_all();`


						resume_sound:
							Resumes the sound.

							`bool sound_pool::resume_sound(int slot);`

						Arguments:
							* int slot: the sound slot you wish to resume.

						Returns:
							bool: true if the sound was resumed, false otherwise.


						sound_is_active:
							Determine whether the sound is active.

							`bool sound_pool::sound_is_active(int slot);`

						Arguments:
							* int slot: the sound slot you wish to check.

						Returns:
							bool: true if the sound is active, false otherwise.

						Remarks:
							If the looping parameter is set to true and the sound object is inactive, the sound is still considered to be active as this just means that we are currently out of earshot. A non-looping sound that has finished playing is considered to be dead, and will be cleaned up.


						sound_is_playing:
							Determine whether the sound is playing.

							`bool sound_pool::sound_is_playing(int slot);`

						Arguments:
							* int slot: the sound slot you wish to check.

						Returns:
							bool: true if the sound is playing, false otherwise.


						update_listener_1d:
							Updates the listener coordinate in 1 dimension.

							`void sound_pool::update_listener_1d(float listener_x);`

						Arguments:
							* float listener_x: the X coordinate of the listener.


						update_listener_2d:
							Updates the listener coordinate in 2 dimensions.

							`void sound_pool::update_listener_2d(float listener_x, float listener_y, double rotation = 0.0);`

						Arguments:
							* float listener_x: the X coordinate of the listener.
							* float listener_y: the Y coordinate of the listener.
							* double rotation = 0.0: the rotation to use.


						update_listener_3d:
							Updates the listener coordinate in 3 dimensions.

							1. `void sound_pool::update_listener_3d(float listener_x, float listener_y, float listener_z, double rotation = 0.0, bool refresh_y_is_elevation = true);`
							2. `void sound_pool::update_listener_3d(vector listener, double rotation = 0.0, bool refresh_y_is_elevation = true);`

						Arguments (1):
							* float listener_x: the X coordinate of the listener.
							* float listener_y: the Y coordinate of the listener.
							* float listener_z: the Z coordinate of the listener.
							* double rotation = 0.0: the rotation to use.
							* bool refresh_y_is_elevation = true: toggles whether `y_is_elevation` property should refresh from the `sound_pool_default_y_is_elevation` global property.

						Arguments (2):
							* vector listener: the coordinates of the listener in vector form.
							* double rotation = 0.0: the rotation to use.
							* bool refresh_y_is_elevation = true: toggles whether `y_is_elevation` property should refresh from the `sound_pool_default_y_is_elevation` global property.


						update_sound_1d:
							Updates the sound coordinate in 1 dimensions.

							`bool sound_pool::update_sound_1d(int slot, int x);`

						Arguments:
							* int slot: the slot of the sound you wish to update.
							* int x: the X coordinate of the sound.

						Returns:
							bool: true if the sound is updated successfully, false otherwise.


						update_sound_2d:
							Updates the sound coordinate in 2 dimensions.

							`bool sound_pool::update_sound_2d(int slot, int x, int y);`

						Arguments:
							* int slot: the slot of the sound you wish to update.
							* int x: the X coordinate of the sound.
							* int y: the Y coordinate of the sound.

						Returns:
							bool: true if the sound is updated successfully, false otherwise.


						update_sound_3d:
							Updates the sound coordinate in 3 dimensions.

							1. `bool sound_pool::update_sound_3d(int slot, int x, int y, int z);`
							2. `bool sound_pool::update_sound_3d(int slot, vector coordinate);`

						Arguments (1):
							* int slot: the slot of the sound you wish to update.
							* int x: the X coordinate of the sound.
							* int y: the Y coordinate of the sound.
							* int z: the Z coordinate of the sound.

						Arguments (2):
							* int slot: the slot of the sound you wish to update.
							* vector coordinate: the coordinate of the sound in vector form.

						Returns:
							bool: true if the sound is updated successfully, false otherwise.


						update_sound_range_1d:
							Updates the sound range in 1 dimensions.

							`bool sound_pool::update_sound_range_1d(int slot, int left_range, int right_range);`

						Arguments:
							* int slot: the slot of the sound you wish to update.
							* int left_range: the left range to update.
							* int right_range: the right range to update.

						Returns:
							bool: true if the sound is updated successfully, false otherwise.


						update_sound_range_2d:
							Updates the sound range in 2 dimensions.

							`bool sound_pool::update_sound_range_2d(int slot, int left_range, int right_range, int backward_range, int forward_range);`

						Arguments:
							* int slot: the slot of the sound you wish to update.
							* int left_range: the left range to update.
							* int right_range: the right range to update.
							* int backward_range: the backward range to update.
							* int forward_range: the forward range to update.

						Returns:
							bool: true if the sound is updated successfully, false otherwise.


						update_sound_range_3d:
							Updates the sound range in 3 dimensions.

							`bool sound_pool::update_sound_range_3d(int slot, int left_range, int right_range, int backward_range, int forward_range, int lower_range, int upper_range, bool update_sound = true);`

						Arguments:
							* int slot: the slot of the sound you wish to update.
							* int left_range: the left range to update.
							* int right_range: the right range to update.
							* int backward_range: the backward range to update.
							* int forward_range: the forward range to update.
							* int lower_range: the bottom / lower range to update.
							* int upper_range: the above / upper range to update.
							* bool update_sound = true: toggles whether all the sound will be updated automatically.

						Returns:
							bool: true if the sound is updated successfully, false otherwise.


						update_sound_start_values:
							Updates the sound start properties.

							`bool sound_pool::update_sound_start_values(int slot, float start_pan, float start_volume, float start_pitch);`

						Arguments:
							* int slot: the slot of the sound you wish to update.
							* float start_pan: the new pan to update.
							* float start_volume: the new volume to update.
							* float start_pitch: the new pitch to update.

						Returns:
							bool: true if the sound is updated successfully, false otherwise.


					Properties:
						behind_pitch_decrease:
							The pitch step size to decrease when the sound is behind. Default is 0.25.

							`float sound_pool::behind_pitch_decrease;`


						max_distance:
							The maximum distance that the sounds can be heard. Default is 0 (disabled).

							`int sound_pool::max_distance;`


						pan_step:
							The pan step size, the sound's final pan or position in HRTF is calculated by multiplying the player's distance from the sound by this value. Default is 1.0.

							`float sound_pool::pan_step;`


						volume_step:
							The volume step size, the sound's final volume is subtracted by this number multiplied by the player's distance from the sound. Default is 1.0.

							`float sound_pool::volume_step;`


						y_is_elevation:
							Toggles whether the Y coordinate is the same as Z, or in otherwords whether the y coordinate determines up/down or forward/backwards in your game world. By default, it retrieves from the `sound_pool_default_y_elevation` global property.

							`bool sound_pool::y_is_elevation = sound_pool_default_y_elevation;`


		Menu Interface (menu.nvgt):
			Classes:
				menu:
					This class gives you an easy way to create a typical menu based on the audio_form list control item, including some nice aditional options such as navigation sounds, wrapping, and more.

					Methods:
						add_item:
							Add an item to the menu.

							`bool menu::add_item(string text, string id = "", int position = -1);`

						Arguments:
							* string text: the text of the item to add to the menu.
							* string id = "": the ID of the item.
							* int position = -1: the position to insert the new item at (-1 = end of menu).

						Returns:
							int: the  position of the new item in the menu.


						intro:
							Produces the intro sequence for the menu. It is not  required to call this function if you don't want to.

							`void menu::intro();`


						monitor:
							Monitors the menu; handles keyboard input, the callback, sounds and more. Should be called in a loop as long as the menu is active.

							`bool menu::monitor();`

						Returns:
							bool: true if the menu should keep running, or false if it has been exited or if an option has been selected.


						reset:
							Resets the menu to it's default state.

							`void menu::reset(bool reset_settings = false);`

						Arguments:
							* bool reset_settings = false: If this is enabled, the sounds and other such properties are cleared.


					Properties:
						automatic_intro:
							If this is true, the intro function will be automatically called the first time the monitor method is invoqued, then the variable will be set to false. It can be set to true again at any time to cause the intro sequence to repeat, or the intro function can be called manually.

							bool menu::automatic_intro;`


						click_sound:
							The sound played when the user changes positions in the menu.

							`string menu::click_sound;`


						close_sound:
							The sound played when the user escapes out of the menu.

							`string menu::close_sound;`


						edge_sound:
							The sound which plays when the user attempts moving beyond the border of a menu while wrapping is disabled.

							`string menu::edge_sound;`


						focus_first_item:
							If this is false, the user will be required to press an arrow key to focus either the first or last item of the menu after the intro function has been called. Otherwise they will be focused on the first item.
							This is disabled by default.

							`bool menu::focus_first_item;`


						intro_text:
							The text spoken when the void intro() function is called.

							`string menu::intro_text;`


						open_sound:
							The sound played when the void intro() function is called.

							`string menu::open_sound;`


						pack_file:
							Optionally set this to a pack containing sounds.

							`pack menu::pack_file;`


						select_sound:
							The sound played when the user chooses an option in the menu.

							`string menu::select_sound;`


						wrap:
							If the user moves beyond an edge of the menu, set this to true to jump them to the other edge, or false to play an edge sound and repeat the last item.
							By default this is set to false.

							`bool menu::wrap;`


						wrap_delay:
							How much time (in ms) should the menu block when wrapping. Defaults to 10ms.

							`uint menu::wrap_delay;`


						wrap_sound:
							The sound played when the menu wraps (only happens when bool wrap = true).

							`string menu::wrap_sound;`


		Music System (music.nvgt):
			Music System:
				This is an include that allows you to easily control music in your game. It can play anything from a stinger to a full music track, control looping, automatic playing, etc.


			classes:
				music_manager:
					methods:
						loop:
							Updates the state of the music manager. This has to be called in your main loop, probably with the value of `ticks()` passed to it.

							`void music_manager::loop(uint64 t);`

						Arguments:
							* uint64 t: the current tick count (you can most likely just make this parameter `ticks()`).


						play:
							Play a music track with particular parameters.

							`bool music_manager::play(string track);`

						Parameters:
							* string track: the music track to play, including any options (see remarks for more information about option syntax).

						Returns:
							bool: true if the track was able to start playing, false otherwise.

						Remarks:
							Music tracks are specified using a very simple string format ("filename; flag; flag; option1=value; option2=value; flag; option3=value...").

							Options are delimited by "; " excluding quotes. The only required setting is the track's main filename, which must be the first option provided. It does not matter in what order any other options or flags are set, the only rule is that the track's configuration string must start with it's main filename.

							The difference between a flag and an option is that a flag is usually just a simple switch E. "stinger; ", while an option usually consists of a key/value pair e. "startpos=2.9"

							When dealing with fades, values are usually in milliseconds, while when dealing with track durations they are usually specified in float seconds E.G. 4.555 for 4555 milliseconds.

						List of possible options:
							* stinger; disables looping
							* loop; causes the track to loop the main file
							* repeat_intro; If this, intro, and repeat are specified, causes the intro track to play again at the repeat point rather than the main track
							* instaplay; If a track is previously playing, causes this one to instantly begin playing instead of the default behavior of fading out and delaying the playback of this track (see switch_predelay and switch_f below), this sets those 2 variables to 0.
							* f=ms; causes the track to fade in at first play, ms=number of milliseconds the fade should take to complete
							* p=pitch; Sets the pitch of this track, defaults to 100. If this track streams from the internet, make sure not to set this too high as this could result in playing more of the sound than has been downloaded.
							* v=volume; sets the starting volume for this entire track (0=full -100=silent)
							* intro=filename; causes the audio file specified here to play before the main track file. By default the main track will begin playing immediately after the intro track ends unless intro_end is specified.
							* repeat=s; How many seconds before the end of the main track should the track repeat again? s=number of seconds (can include milliseconds like 4.681). When the track repeats, it will play overtop the remainder of the currently ending track. repeat=0 is the same as the loop flag, repeat=anything<0 is the same as stinger. If multiple of repeat, loop, stinger are specified, only the one specified last will take effect.
							* repeat_f=ms; If repeat is specified and is greater than 0, this option causes the end of the currently playing track to fade out as the repeated track begins to play, and specifies how many ms that fade should take to complete.
							* predelay=s; how many seconds (can include milliseconds) before the track should begin after it starts playing?
							* switch_predelay=s; Same as above, but only applies if a track was previously playing and the music system switches to this one. Defaults to 300. This and predelay are added together, but this variable is handled by the music manager instead of by this music track unlike predelay.
							* switch_f=ms; If the music system is playing a track and then switches to this one, how many milliseconds should it take for the previously playing track to fade out? Defaults to 400.
							* intro_end=s; if intro is specified but the audio track specified by intro does not seamlessly transition into the main track, how many seconds (can include milliseconds) before the intro ends should the main track begin playing? The main track will not interrupt the remainder of the intro but will play overtop of it if this option is specified.
							* startpos=s; How many seconds (can include milliseconds) into either the intro track if specified else the main track should the audio begin playing E. seek?


						set_load_callback:
							This system was originally made for Survive the Wild which needs to read sound data from strings not packs most of the time, so this class implements something more complicated than a music.pack variable being set. Someone feel free to add this functionality though or I may do later. Instead, we set a callback which receives a sound object and a filename, and calls the appropriate load method on that sound for your situation. Not needed if your sounds are simply on disk. A short example of a load callback is below.

							`void music_manager::set_load_callback(load_music_sound@ cb);`

						Arguments:
							* load_music_sound@ cb: your load callback. The syntax of it is `sound@ load_music_sound(sound@ sound_to_load, string filename_to_load);`. See remarks for an example.

						Remarks:
							This is a basic example of how to write and set up a sound loading callback for use with the music manager.

							sound@ music_load(sound@ sound_to_load, string filename_to_load) {
								// Usually a sound object will be provided to this function from the music manager. Encase not,
								if (@sound_to_load is null) @sound_to_load = sound();
								if( !sound_to_load.load(filename_to_load, pack_file)) return null; // indicates error.
								return s; // Return a handle to the loaded sound.
							}

							// ...

							your_music_manager.set_load_callback(music_load);


						stop:
							Stops any currently playing music.

							`void music_manager::stop(int fade = 0);`

						Arguments:
							* int fade = 0: the fade duration of the currently playing music (in milliseconds).


					properties:
						playing:
							Determine if the music manager is currently playing a track or not.

							`bool music_manager::playing;`


						volume:
							Controls the volume of any currently playing music in this music manager.

							`float music_manager::volume;`


		Number Speaking (number_speaker.nvgt):
			classes:
				number_speaker:

					The `number_speaker` object is used to facilitate natural and smooth number speech. It is best to use this class if you're developing a game with voice acting.

					number_speaker();

				Remarks:

					The `number_speaker` class searches for sound files that match the spoken components of a number as closely as possible, separating words with an underscore. For instance, if the number is 72, it'll first check for a file named seventy_two.wav. If this file is not available, it'll search for seventy.wav and two.wav and combine them.

					This system allows you to record as many number sounds as desired to achieve flawless spoken output. Alternatively, you can record only essential numbers, such as zero.wav to nineteen.wav, twenty.wav to ninety.wav, and optionally hundred.wav, thousand.wav, and million.wav. You can also record a file named minus.wav to distinguish between positive and negative numbers, as well as and.ogg to have the word "and" spoken if the include_and parameter is enabled.

				Example:

					```NVGT
				include "number_speaker.nvgt":
					void main() {
						// Speak a number using speak_wait.
						number_speaker test;
						test.speak_wait(350);
					}



					methods:
						speak:

							This method will speak a number.

							int number_speaker::speak(double the_number);

						Arguments:

							* double the_number: the number to speak.

						Returns:

							int: 0 on success, -1 on failure.

						Remarks:

							The speak method searches for sound files based on the values stored in the prepend and append properties, determining the most appropriate sound files to use by performing multiple searches based on the given number.

							When this function is called, it returns immediately, allowing the script to continue running while the number is being spoken. To ensure that numbers are spoken smoothly and fluently, it is necessary to call the speak_next method continuously.

						Example:

							```NVGT
							#include "number_speaker.nvgt"
							void main() {
								number_speaker number;
								number.speak(350);
								while(number.speak_next()==1) {
									wait(5);
								}
							}



						speak_next:

							This method is used to check the status of a number that is currently being spoken or has already been spoken, and, if necessary, initiates the playback of the next number.

							int number_speaker::speak_next();

						Returns:

							int: 0 if there are no more files to be played, 1 if there are more files to be played or if the last file is still playing, and -1 if an error occurs.

						Remarks:

							The speak_next method works in conjunction with the speak method and should be used continuously while the number is being spoken.

							This method functions not only as a status indicator but also as a trigger, checking and playing numbers as needed. To ensure that the numbers are spoken smoothly and fluently, it is essential to perform this check at regular intervals, approximately every 5 milliseconds.

						Example:

							```NVGT
							#include "number_speaker.nvgt"
							void main() {
								// Speak a number using speak and speak_next.
								number_speaker test;
								test.speak(350);
								while(test.speak_next()==1) {
									wait(5);
								}
							}



						speak_wait:

							This method will speak a number and wait until the number is fully read before returning.

							int number_speaker::speak_wait(double the_number);

						Arguments:

							* double the_number: the number to speak.

						Returns:

							int: 0 on success, -1 on failure.

						Remarks:

							The speak_wait method searches for sound files based on the values stored in the prepend and append properties, determining the most appropriate sound files to use by performing multiple searches based on the given number.

							The speak_wait function pauses execution until the number has been fully spoken before returning. This means that the script execution is halted for the duration of the reading. To avoid this, you can use the speak and speak_next methods instead.

						Example:

							```NVGT
							#include "number_speaker.nvgt"
							void main() {
								// Speak a number using speak_wait.
								number_speaker test;
								test.speak_wait(350);
							}



						stop:

							This method stops any number that is currently being spoken.

							void number_speaker::stop();

						Example:

							```NVGT
							#include "number_speaker.nvgt"
							void main() {
								// Speak a number using speak and speak_next, stopping it prematurely if the user presses space.
								number_speaker test;
								test.speak(350);
								while(test.speak_next()==1) {
									if(key_pressed(KEY_SPACE))
										test.stop();
									wait(5);
								}
							}



						set_sound_object:

							This method allows you to specify an existing sound object to be used for any auditory feedback in the number speaker.

							bool number_speaker::set_sound_object(sound@ handle);

						Arguments:

							* sound@ handle: a reference to an existing sound object that will be utilized for all future sound output.

						Returns:

							bool: true on success, false on failure.

						Remarks:

							The object specified in this function will handle all future auditory feedback provided by the number speaker. If this method is not called, an internal sound object will be used by default for sound output.

						Example:

							```NVGT
							#include "number_speaker.nvgt"
							void main() {
								sound test;
								number_speaker number;
								number.set_sound_object(test);
								number.speak_wait(350);
							}



					properties:
						append:
							This string will be appended to any loaded number file. The default extension is .wav, so you only need to change this if your number files use a different extension.

							`string number_speaker::append;`


						prepend:
							This string will be prepended to any loaded number file. It is useful if your number files are stored in a separate directory or if the filenames include a specific prefix.

							`string number_speaker::prepend;`


						include_and:
							This boolean determines whether the word "and" should be included in the appropriate places in the output. If set to true, you must create a file named and.wav, unless the extension (e.g., the append property) is something other than .wav. The default value is false.

							`bool number_speaker::include_and;`


						pack_file:
							The pack file to use in the object.

							`pack_file@ number_speaker::pack_file = null;`


		Size Conversions (size.nvgt):
			Size Conversions Include:
				This include provides a simple function to convert an unsigned int to a human readable size. It also provides a few very useful size constants.


			Functions:
				size_to_string:

					Converts an unsigned int into a human-readable size.

					string size_to_string(uint64 size, uint8 round_place = 2);

				Arguments:

					* uint64 size: the size to convert.

					* uint8 round_place = 2: how many decimal places to round the sizes to.

				Returns:

					The given size as a human-readable string.

				Example:

					```NVGT
					#include "size.nvgt"
					void main() {
						uint64[] sizes = {193, 3072, 1048576, 3221225472, 1099511627776, 35184372088832};
						string results;
						for (uint i = 0; i < sizes.length(); i++)
							results += sizes[i] + " bytes = " + size_to_string(sizes[i]) + ",\n";
						// Strip off the trailing comma and new line.
						results.trim_whitespace_right_this();
						results.erase(results.length() - 1);
						alert("Results", results);
					}



			Global Properties:
				GIGABYTES:

					The number of bytes in a gigabyte.

					const uint GIGABYTES;

				Example:

					```NVGT
					#include "size.nvgt"
					void main() {	
						alert("Info", "A gigabyte is " + GIGABYTES + " bytes");
					}



				KILOBYTES:

					The number of bytes in a kilobyte.

					const uint KILOBYTES;

				Example:

					```NVGT
					#include "size.nvgt"
					void main() {
						alert("Info", "10 kilobytes is " + 10 * KILOBYTES + " bytes");
					}



				MEGABYTES:

					The number of bytes in a megabyte.

					const uint MEGABYTES;

				Example:

					```NVGT
					#include "size.nvgt"
					void main() {
						alert("Info", "19 megabytes is " + 19 * MEGABYTES + " bytes");
					}



				SIZE_TO_STRING_UNITS:

					String array of all the supported units for size_to_string().

					const array<string> SIZE_TO_STRING_UNITS;

				Example:

					```NVGT
					#include "size.nvgt"
					void main() {
						string possible_units;
						for (uint i = 0; i < SIZE_TO_STRING_UNITS.length(); i++)
							possible_units += SIZE_TO_STRING_UNITS[i] + ",\n";
						// Strip off the trailing comma and new line.
						possible_units.trim_whitespace_right_this();
						possible_units.erase(possible_units.length() - 1);
						alert("Info", "The possible units are: " + possible_units + ".");
					}



				TERABYTES:

					The number of bytes in a terabyte.

					const uint TERABYTES;

				Example:

					```NVGT
					#include "size.nvgt"
					void main() {	
						alert("Info", "A terabyte is " + TERABYTES + " bytes");
					}



		Statistic Management (stat_set.nvgt):
			Classes:
				stat_set:
					Methods:
						add:
							Adds a stat to the set.

							`stat@ stat_set::add(const string&in name, var@ value, const string&in text = "", stat_callback@ callback = null, dictionary@ user = null);`

						Arguments:
							* const string&in name: the name of the stat to add.
							* var@ value: the starting value for this stat.
							* const string&in text = "": an optional text template to be used when getting the value of this stat. In this template, use %0 anywhere you want to output the raw value of the stat itself.
							* stat_callback@ callback = null: an optional callback to call every time the value of this stat is requested. One of text or callback has to be provided in order for a stat to work correctly.
							* dictionary@ user = null: Not touched by the stat_set itself, allows the linkage of any user values to a stat to help with display.

						Returns:
							stat@: a handle to the newly added stat object.


						delete:
							Deletes a stat from the set.

							`bool stat_set::delete(const string&in name);`

						Arguments:
							* const string&in name: the name of the stat to be deleted.

						Returns:
							bool: true if the stat was successfully located and deleted, false otherwise.


						exists:
							Determine if a stat with the given name exists in the set.

							`bool stat_set::exists(const string&in stat_name) const;`

						Arguments:
							* const string&in stat_name: name of the statt to check for.

						Returns:
							bool: true if a stat with the specified name exists in the `stat_set`, false otherwise.


						reset:
							Clears all stats from the set.

							`void stat_set::reset();`


						update:
							Updates a particular stat with a new value.

							`void stat_set::update(const string&in name, var@ value);`

						Arguments:
							* const string&in name: the name of the stat to be updated.
							* var@ value: the new value of the stat.


					Operators:
						opIndex:
							Allows for the easy access of stats with `[...]` syntax.

							`stat@ stat_set::opIndex(const string&in stat_name) const;`

						Arguments:
							* const string&in stat_name: the name of the stat you want to access.

						Returns:
							stat@: a handle to a stat with the given name, or null if no stat could be found.


					Properties:
						size:
							Returns the number of stats in the set.

							`int stat_set::size;`


		Token Generation (token_gen.nvgt):
			Token generation include:
				Allows you to easily generate random strings of characters of any length in a given mode, and possibly custom function if you want to generate only certain characters.


			Enums:
				token_gen_flag:
					This enum holds various constants that can be passed to the mode parameter of the generate_token function in order to control what characters appear in results.

					* TOKEN_CHARACTERS = 1: Allows for characters, a-zA-Z
					* TOKEN_NUMBERS = 2: Allows for numbers, 0-9
					* TOKEN_SYMBOLS = 4: Uses only symbols, \`\~\!\@\#\$\%\^\&\*\(\)\_\+\=\-\[\]\{\}\/\.\,\;\:\|\?\>\<

				Remarks:
					These are flags that should be combined together using the bitwise OR operator. To generate a token containing characters, numbers and symbols, for example, you would pass `TOKEN_CHARACTERS | TOKEN_NUMBERS | TOKEN_SYMBOLS` to the mode argument of generate_token. The default flags are `TOKEN_CHARACTERS | TOKEN_NUMBERS`.


			Functions:
				generate_token:

					Generates a random string of characters (a token).

					string generate_token(int token_length, int mode = TOKEN_CHARACTERS | TOKEN_NUMBERS)

				Arguments:

					* int token_length: the length of the token to generate.

					* int mode = TOKEN_CHARACTERS | TOKEN_NUMBERS: allows you to specify which characters you would like in the generated token (see remarks).

				returns:

					String: a random token depending on the modes.

				Remarks:

					This function uses the `generate_custom_token` function to generate. The characters used to generate the token will depend on the modes you specified. See `token_gen_flags` enum constants for a list of supported modes that can be passed to the mode parameter.

				Example:

					```NVGT
					#include "token_gen.nvgt"
					void main() {
						alert("Info", "Your token is: " + generate_token(10));
						alert("Info", "Numbers only token is: " + generate_token(10, TOKEN_NUMBERS));
						alert("Info", "Characters only token is: " + generate_token(10, TOKEN_CHARACTERS));
					}



				generate_custom_token:

					Generates a random string of characters (a token) while optionally allowing you to directly specify the characters you wish to use in the generation.

					1. string generate_custom_token(int token_length, string[] values);

					2. string generate_custom_token(int token_length, string characters);

				Arguments (1):

					* int token_length: the length of the token to generate.

					* string[] values: an array of possible choices (a list of characters or even words) to generate from.

				Arguments (2):

					* int token_length: the length of the token to generate.

					* string characters: a list of characters to generate.

				returns:

					String: a random token.

				Remarks:

					If the `values` array or `characters` string is empty or `token_length` is set to 0 or less, an empty string is returned.

				Example:

					```NVGT
					#include "token_gen.nvgt"
					void main() {
						alert("Info", "Your A to C token is: " + generate_custom_token(10, {"a", "b", "c"}));
						alert("Info", "A to C with capitals included token is: " + generate_custom_token(10, {"a", "b", "c", "A", "B", "C"}));
					}



		touch gesture management (touch.nvgt):
			Classes:
				touch_gesture_manager:

					Detects and then converts raw touch screen finger data into gesture events which can be detected by your application or even converted to keyboard input.

					touch_gesture_manager();

				Remarks:

					This is the highest level interface NVGT offers to accessing touch screens. It looks at what fingers are touching the screen and where, and by monitoring that data over time, derives gestures such as swipes and taps from that data before passing them along to whatever interface you register to receive them.

					An interface is included by default (see the touch_keyboard_interface class), which convertes these events into simulated keyboard input for you so that adding touch support to your game becomes as simple as mapping gesture names to keycode lists.

					The basic usage of this class is to create an instatnce of it somewhere, before attaching any number of touch_interface instances to the manager and finally calling manager.monitor() within the loops of your programs to receive gesture events.

				Example:

					```NVGT
				include"touch.nvgt":
				include"speech.nvgt":
					touch_gesture_manager touch;
					void main() {
						show_window("touch screen test");
						//attach multiple keys to a single gesture.
						int[] t = {KEY_TAB, KEY_LSHIFT};
						touch_keyboard_interface testsim(touch, {{"swipe_left1f", KEY_LEFT}, {"swipe_right1f", KEY_RIGHT}, {"swipe_up1f", KEY_UP}, {"swipe_down1f", KEY_DOWN}, {"double_tap1f", KEY_RETURN}, {"tripple_tap1f", t}, {"swipe_left2f", KEY_ESCAPE}, {"swipe_right2f", KEY_Q}, {"double_tap2f", KEY_SPACE}});
						touch.add_touch_interface(testsim);
						dictionary@d = {{"double_tap1f", KEY_X}, {"double_tap2f", KEY_Z}};
						// Demonstrate registering an interface for only a portion of the screen.
						touch.add_touch_interface(touch_keyboard_interface(touch, d, 0.5, 1.0, 0.5, 1.0));
						while (!key_pressed(KEY_ESCAPE) and !key_pressed(KEY_AC_BACK)) {
							wait(5);
							touch.monitor();
							if (key_pressed(KEY_LEFT)) speak("left");
							if (key_pressed(KEY_RIGHT)) speak("right");
							if (key_pressed(KEY_DOWN)) speak("down");
							if (key_pressed(KEY_UP)) speak("up");
							if (key_pressed(KEY_TAB)) {
								speak("tripple tap");
								if (key_down(KEY_LSHIFT) or key_down(KEY_RSHIFT)) speak("you are holding down shift", false);
							}
							if (key_pressed(KEY_SPACE)) speak("double tap 2 fingers");
							if (key_pressed(KEY_ESCAPE)) exit();
							if (key_pressed(KEY_Q)) speak("swipe right 2 fingers");
							if (key_pressed(KEY_RETURN)) speak("you just double tapped the screen!", false);
							if (key_pressed(KEY_X)) speak("You just double tapped another part of the screen!");
							if (key_pressed(KEY_Z)) speak("You just double tapped with 2 fingers on another part of the screen!");
						}
					}



					Methods:
						add_touch_interface:
							Adds a single interface to the list of interfaces that will receive touch events.

							```bool add_touch_interface(touch_interface@ interface);```

						Arguments:
							* touch_interface@ interface: The interface to add to the manager.

						Returns:
							bool: True if the interface was successfully added, false otherwise.

						Remarks:
							You can add multiple interfaces to a gesture manager because different interfaces can receive different events for various parts of the screen. An interface can simply return false in its is_bounds method at any time to pass the gesture event to the next handler in the chain. Gesture interfaces are evaluated from newest to oldest.


						clear_touch_interfaces:
							Removes all interfaces from the manager.

							```void clear_touch_interfaces();```

						Remarks:
							This method clears all registered touch interfaces, effectively disabling all gesture handling until new interfaces are added.


						is_available:

							Determine whether any touch devices are available to receive events from.

							bool is_available();

						Returns:

							bool: true if at least one touch device is available on the system, false otherwise.

						Remarks:

							It is worth noting that due to circumstances outside our control, sometimes touch devices don't appear in the list until after you have touched them at least once. Therefor you should not use this method at program startup to determine with finality that no touch device is available, but could instead use it during program lifetime to monitor for whether touch support appears or disappears.
							```NVGT
							#include "touch.nvgt"
							// Example:
							touch_gesture_manager touch;
							void main() {
								wait(1000); // Give the user some time to touch the screen to make sure it appears.
								if (!touch.is_available()) alert("warning", "This system does not appear to support touch screen devices");
							}



						monitor:
							Check for the latest touch events.

							```void monitor();```

						Remarks:
							This function must be called in all of your game's loops in order for the system to work properly. Not doing this will lead to undefined behavior.

						set_touch_interfaces:
							Sets the list of interfaces that will receive touch events.

							```bool set_touch_interfaces(touch_interface@[] interfaces, bool append = false);```

						Arguments:
							* touch_interface@[]@ interfaces: A list of interfaces that will receive touch events.
							* bool append = false: Determines whether to append to the list of existing interfaces vs. replacing it.

						Returns:
							bool: True if a change was made to the interface list, false otherwise.

						Remarks:
							You can pass multiple interfaces to a gesture manager because different interfaces can receive different events for various parts of the screen. An interface can simply return false in its is_bounds method at any time to pass the gesture event to the next handler in the chain. Gesture interfaces are evaluated from newest to oldest.


					Properties:
						flick_velocity_threshold:
							The minimum velocity (normalized distance per second) required to trigger a flick gesture instead of a regular swipe.

							```float flick_velocity_threshold = 1.5f;```

						Remarks:
							Flick gestures are faster versions of swipes that can trigger different actions in your interface.


						hold_jitter_threshold:
							The maximum distance (normalized 0.0-1.0) a finger can move while still being considered stationary for long press detection.

							```float hold_jitter_threshold = 0.05f;```

						Remarks:
							If a finger moves more than this distance from its starting position, long press and hold events will not trigger.


						long_press_time:
							The minimum time in milliseconds a finger must remain stationary to trigger a long press.

							```uint long_press_time = 600;```

						Remarks:
							Default value is 600 milliseconds. The finger must not move beyond the hold_jitter_threshold during this time.


						multi_tap_timeout:
							The maximum time in milliseconds between consecutive taps to be considered part of a multi-tap sequence.

							```uint multi_tap_timeout = 600;```

						Remarks:
							Default value is 600 milliseconds. This controls how quickly users must perform double-taps, triple-taps, etc.


						swipe_min_dist:
							The minimum distance (normalized 0.0-1.0) a finger must travel to be considered a swipe.

							```float swipe_min_dist = 0.1f;```

						Remarks:
							This value is in normalized coordinates where 1.0 represents the full width or height of the screen.


						swipe_segment_threshold:
							The minimum distance (normalized 0.0-1.0) before detecting a new swipe direction in a compound swipe.

							```float swipe_segment_threshold = 0.1f;```

						Remarks:
							This controls how sensitive the system is to direction changes during compound swipes (e.g., L-shaped swipes).


						tap_max_delay:
							The maximum time in milliseconds between a finger going down and up to be considered a tap.

							```uint tap_max_delay = 250;```

						Remarks:
							Default value is 250 milliseconds. Increase this if users are having trouble registering taps.


				touch_interface:
					Base class for creating custom touch gesture handlers.

					```touch_interface(touch_gesture_manager@ parent, float minx = TOUCH_UNCOORDINATED, float maxx = TOUCH_UNCOORDINATED, float miny = TOUCH_UNCOORDINATED, float maxy = TOUCH_UNCOORDINATED);```

				Arguments:
					* touch_gesture_manager@ parent: A handle to the manager this interface will be attached to.
					* float minx, maxx, miny, maxy = TOUCH_UNCOORDINATED: The bounds of this interface in normalized coordinates (0.0-1.0), default is entire screen.

				Remarks:
					This is the base class that all touch interfaces inherit from. You can extend this class to create custom gesture handlers with specialized behavior. The class provides virtual methods for all gesture types that you can override in derived classes.

					Any method starting with the word "on_" that you see in the methods list for this base class is meant to be overwridden, such as on_long_press, on_double tap etc.


					Methods:
						on_compound_swipe:
							Handles compound swipe gestures.

							```void on_compound_swipe(touch_screen_finger@ finger, int[] directions);```

						Arguments:
							* touch_screen_finger@ finger: The finger that performed the compound swipe.
							* int[] directions: Array of direction values representing the path of the swipe.

						Remarks:
							Compound swipes occur when a finger changes direction during a single swipe gesture, such as an L-shaped or Z-shaped motion.


						on_double_tap:
							Handles double tap gestures.

							`void on_double_tap(touch_screen_finger@ finger, uint finger_count);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.
							* uint finger_count: The number of fingers that this gesture consists of.


						on_flick:
							Handles flick gestures.

							`void on_flick(touch_screen_finger@ finger, int direction, float velocity);`

						Arguments:
							* touch_screen_finger@ finger: The finger that performed the flick.
							* int direction: The direction of the flick (see swipe_touch_directions enum).
							* float velocity: The velocity of the flick in normalized units per second.

						Remarks:
							Flicks are detected when a swipe's velocity exceeds `flick_velocity_threshold`.


						on_hold:
							Handles hold gestures.

							`void on_hold(touch_screen_finger@ finger, uint finger_count);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.
							* uint finger_count: The number of fingers that this gesture consists of.

						Remarks:
							Triggered continuously while a finger continues to be held down after a long press has been detected.


						on_long_press:
							Handles long press gestures.

							`void on_long_press(touch_screen_finger@ finger, uint finger_count);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.
							* uint finger_count: The number of fingers that this gesture consists of.

						Remarks:
							Triggered once when a finger has been held down for longer than `long_press_time` without moving beyond `hold_jitter_threshold`.


						on_released_finger:
							Called when a finger is lifted from the screen.

							`void on_released_finger(touch_screen_finger@ finger);`

						Arguments:
							* touch_screen_finger@ finger: The finger that was just released.

						Remarks:
							This is useful for cleanup operations, such as releasing simulated keys.


						on_single_tap:
							Handles single tap gestures.

							`void on_single_tap(touch_screen_finger@ finger, uint finger_count);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.
							* uint finger_count: The number of fingers that this gesture consists of.


						on_swipe_down:
							Handles swipe down gestures.

							`void on_swipe_down(touch_screen_finger@ finger);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.



						on_swipe_down_left:
							Handles swipe down-left gestures.

							`void on_swipe_down_left(touch_screen_finger@ finger);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.

						Remarks:
							These diagonal swipe methods are only triggered when `touch_enable_8_way_swipes` is set to true.


						on_swipe_down_right:
							Handles swipe down-right gestures.

							`void on_swipe_down_right(touch_screen_finger@ finger);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.

						Remarks:
							These diagonal swipe methods are only triggered when `touch_enable_8_way_swipes` is set to true.


						on_swipe_left:
							Handles swipe left gestures.

							`void on_swipe_left(touch_screen_finger@ finger);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.


						on_swipe_right:
							Handles swipe right gestures.

							`void on_swipe_right(touch_screen_finger@ finger);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.


						on_swipe_up:
							Handles swipe up gestures.

							`void on_swipe_up(touch_screen_finger@ finger);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.


						on_swipe_up_left:
							Handles swipe up-left gestures.

							`void on_swipe_up_left(touch_screen_finger@ finger);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.

						Remarks:
							These diagonal swipe methods are only triggered when `touch_enable_8_way_swipes` is set to true.


						on_swipe_up_right:
							Handles swipe up-right gestures.

							`void on_swipe_up_right(touch_screen_finger@ finger);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.

						Remarks:
							These diagonal swipe methods are only triggered when `touch_enable_8_way_swipes` is set to true.


						on_triple_tap:
							Handles triple tap gestures.

							`void on_tripple_tap(touch_screen_finger@ finger, uint finger_count);`

						Arguments:
							* touch_screen_finger@ finger: The primary finger that triggered this gesture.
							* uint finger_count: The number of fingers that this gesture consists of.


					Properties:
						allow_passthrough:
							Determines whether gestures should continue to be processed by other interfaces after this one handles them.

							```bool allow_passthrough = false;```

						Remarks:
							When set to true, gesture events will be passed to the next interface in the chain even after this interface processes them. This allows multiple interfaces to respond to the same gesture.


				touch_keyboard_interface:
					Convert gesture events to simulated keyboard input.

					```touch_keyboard_interface(touch_gesture_manager@ parent, dictionary@ map, float minx = TOUCH_UNCOORDINATED, float maxx = TOUCH_UNCOORDINATED, float miny = TOUCH_UNCOORDINATED, float maxy = TOUCH_UNCOORDINATED);```

				Arguments:
					* touch_gesture_manager@ parent: A handle to the manager you intend to add this interface to.
					* dictionary@ map: A mapping of gestures to keycode lists (see remarks).
					* float minx, maxx, miny, maxy = TOUCH_UNCOORDINATED: The bounds of this interface, default for entire screen or custom.

				Remarks:
					This interface works by receiving a mapping of gesture names or IDs to lists of keycodes that should be simulated.

					The basic format of gesture IDs consists of a gesture name followed by a number of fingers and the letter f. For example, to detect a 2 finger swipe right gesture, you would use the key "swipe_right2f" in the mapping dictionary.

				Available gesture names:

					**Basic swipes (4-way):**
					* swipe_left
					* swipe_right
					* swipe_up
					* swipe_down

					**Diagonal swipes (8-way, requires `touch_enable_8_way_swipes = true`):**
					* swipe_up_left
					* swipe_up_right
					* swipe_down_left
					* swipe_down_right

					**Taps:**
					* single_tap (or just use the numbered versions below)
					* double_tap
					* tripple_tap
					* 4_tap, 5_tap, etc. (for higher tap counts)

					**Advanced gestures:**
					* flick_left, flick_right, flick_up, flick_down (fast swipes)
					* long_press (finger held down without movement)
					* Compound swipes (e.g., "swipe_left_up" for an L-shaped gesture)

				Example gesture mappings:
					```nvgt
					dictionary@ gestures = {
					    {"swipe_left1f", KEY_LEFT},           // 1-finger swipe left
					    {"swipe_right2f", KEY_RIGHT},         // 2-finger swipe right
					    {"double_tap1f", KEY_RETURN},         // Double tap with 1 finger
					    {"tripple_tap2f", KEY_ESCAPE},        // Triple tap with 2 fingers
					    {"long_press1f", KEY_SPACE},          // Long press with 1 finger
					    {"flick_up1f", KEY_PAGEUP}            // Quick flick upward
					};

			Constants:
				TOUCH_UNCOORDINATED:
					const float TOUCH_UNCOORDINATED = -10.0f;

				Remarks:
					Special value used to indicate that a touch interface should respond to the entire screen rather than a specific bounded region.


			Enums:
				swipe_touch_directions:
					enum swipe_touch_directions {
					    swipe_touch_direction_left = 0,
					    swipe_touch_direction_right,
					    swipe_touch_direction_up,
					    swipe_touch_direction_down,
					    swipe_touch_direction_up_left,
					    swipe_touch_direction_up_right,
					    swipe_touch_direction_down_left,
					    swipe_touch_direction_down_right
					}

				Remarks:
					These values represent the possible directions for swipe gestures. The diagonal directions are only used when `touch_enable_8_way_swipes` is enabled.


			Global Properties:
				touch_enable_8_way_swipes:
					bool touch_enable_8_way_swipes = false;

				Remarks:
					When set to true, enables detection of 8-way directional swipes (including diagonals: up_left, up_right, down_left, down_right). When false, only 4-way swipes are detected (up, down, left, right).


		User Data Storage and Retrieval (settings.nvgt):
			Classes:
				settings:
					This class is designed to save and load data from files in various formats, specified by the company and product.

					`settings();`

				Remarks:
					When this object is first constructed, it will not be active. To activate it you must call the `setup` method, specifying the company name, product name, and optionally the format you wish to use. Please see the setup method documentation for more information and a list of supported formats.


					Methods:
						setup:
							This method will setup your company and/or product, and thus the object will be activated, allowing you to interact with the data of the product.

							`bool settings::setup(const string company, const string product, const bool local_path, const string format = "ini");`

						Arguments:
							* const string company: the name of your company. This is the main folder for all your products related to this company.
							* const string product: the name of your product. This will be the subfolder under the company.
							* const bool local_path: toggles whether the data should be saved in the path where the script is being executed, or in the appdata.
							* const string format = "ini": the format you wish to use, see remarks.

						Returns:
							bool: true on success, false on failure.

						Remarks:
							The following is a list of supported formats:
							* `ini`: default format (.ini).
							* `json`: JSON format (.json).
							* `nvgt`: this format will use built-in dictionary (.dat).

							Note that it does not currently allow you to modify the extentions, for instance, .savedata, .sd, etc. In the future it might be possible and will be documented as soon as it gets implemented.


						close:
							This method closes the settings object and therefore the object will be deactivated. From then on you will no longer be able to interact with the data of the company and/or products anymore unless you resetup with `settings::setup` method.

							`bool settings::close(bool save_data = true);`

						Arguments:
							* bool save_data = true: toggles whether the data should be saved before closing.

						Returns:
							bool: true on success, false on failure.


						dump:
							This method will manually save the data.

							`bool settings::dump();`

						Returns:
							bool: true on success, false on failure.

						Remarks:
							Especially if you have the `instant_save` property set to false, you need to call this function to save the data manually. Alternatively, the data will be saved if you close the object with `settings::close` method and set the `save_data` parameter to true, which is default.


						exists:
							Determines whether the key exists in the data.

							`bool settings::exists(const string&in key);`

						Arguments:
							* const string&in key: the key to look for.

						Returns:
							bool: true if the key exists, false otherwise.


						has_other_products:
							Determines whether this company has other products (i.e it has more than 1 products).

							`bool settings::has_other_products();`

						Returns:
							bool: true if the company has more than 1 products, false on failure.


						read_number:
							Reads the data by a given key, number as value.

							`double settings::read_number(const string&in key, double default_value = 0);`

						Arguments:
							* const string&in key: the key to look for.
							* double default_value = 0: the value to return if the key could not be retrieved.

						Returns:
							double: the data of the key or default value if the key could not be retrieved.


						read_string:
							Reads the data by a given key.

							`string settings::read_string(const string&in key, const string&in default_value = "");`

						Arguments:
							* const string&in key: the key to look for.
							* const string&in default_value = "": the value to return if the key could not be retrieved.

						Returns:
							string: the data of the key or default value if the key could not be retrieved.


						remove_product:
							This method removes the current product.

							`bool settings::remove_product();`

						Returns:
							bool: true on success, false on failure

						Remarks:
							This method will delete the directory of the current product if there are other products in the company. Otherwise, the company directory will be deleted.


						remove_value:
							This method removes the key from the data.

							`bool settings::remove_value(const string&in value_name);`

						Arguments:
							* const string&in value_name: the key to remove.

						Returns:
							bool: true on success, false on failure


						write_number:
							this method writes into the data by a given key, number as value.

							`bool settings::write_number(const string&in key, double number);`

						Arguments:
							* const string&in key: the key to write to.
							* double number: the number to write.

						Returns:
							bool: true on success, false on failure


						write_string:
							this method writes into the data by a given key.

							`bool settings::write_string(const string&in key, const string&in value);`

						Arguments:
							* const string&in key: the key to write to.
							* const string&in value: the value to write.

						Returns:
							bool: true on success, false on failure


					Properties:
						active:
							Returns true if the settings object is active (i.e it is possible to use). You cannot use the settings object if this property is false.

							`bool settings::active;`


						company_name:
							The name of your company. This will be the main folder to save all products related to this company.

							`string settings::company_name;`


						encryption_key:
							The key to use if you want the data to be encrypted. By default, no encryption key is specified.

							`string settings::encryption_key;`


						instant_save:
							Toggles whether the data file should be automatically saved as soon as you add the data. true is default option. Turning this off will have to be saved manually using `settings::dump` method.

							`bool settings::instant_save;`


						local_path:
							Toggles whether the data should be saved in the path where the script is being executed, or in the appdata. false is default.

							`bool settings::local_path;`


						product:
							The name of your product.

							`string settings::product;`


	Plugins:
		git2nvgt:
		nvgt_curl:
			classes:
				internet_request:
					properties:
						bytes_downloaded:
							The number of bytes currently downloaded with this internet_request.

							`double internet_request::bytes_downloaded;`


						bytes_uploaded:
							The number of bytes that have been uploaded with this internet_request.

							`double internet_request::bytes_uploaded;`


						complete:
							Determine if the active request has completed yet or not.

							`bool internet_request::complete;`


						download_percent:
							The current percentage downloaded.

							`double internet_request::download_percent;`


						download_size:
							The size of the data you're downloading (in bytes).

							`double internet_request::download_size;`


						follow_redirects:
							Should your request follow HTTP redirects?

							`bool internet_request::follow_redirects;`


						in_progress:
							Determine if the request is currently in progress.

							`bool internet_request::in_progress;`


						max_redirects:
							The maximum number of redirects to perform before giving up.

							`int internet_request::max_redirects;`


						no_curl:
							Tells you if libcurl was successfully able to initialize this request, do not use this object if this property is false!

							`bool no_curl;`


						status_code:
							Represents the HTTP status code returned by this request.

							`int internet_request::status_code;`


						upload_percent:
							The percentage uploaded.

							`double internet_request::upload_percent;`


						upload_size:
							The size of your upload (in bytes).

							`double internet_request::upload_size;`


			functions:
				curl_url_decode:
					Decode an encoded URL using curl.

					`string curl_url_decode(string url);`

				Arguments:
					* string url: the URL to decode.

				Returns:
					string: the decoded URL.

				Remarks:
					This functionality exists natively in NVGT too, the curl functions are just provided here for completeness. For more information, see the built-in `url_decode()` function.


				curl_url_encode:
					Encode a URL using curl.

					`string curl_url_encode(string url);`

				Arguments:
					* string url: the URL to encode.

				Returns:
					string: the encoded URL.

				Remarks:
					This functionality exists natively in NVGT too, the curl functions are just provided here for completeness. For more information, see the built-in `url_encode()` function.


		nvgt_sqlite:
		systemd_notify:

			Wrapper for the sd_notify Linux function which can be useful if you are writing a systemd service. On Windows you can include this plugin and compile your game, this function will do nothing. On Linux though, values will be sent to systemd as described in the sd_notify man page (see remarks) when this function is called.

			int systemd_notify(string state);

		Arguments:

			* string state: the message to send to systemd.

		Returns:

			int: the value returned by sd_notify.

		Remarks:

			For more information about systemd, see its man page: https://man7.org/linux/man-pages/man3/sd_notify.3.html

		Example:

			```NVGT
		pragma plugin systemd_notify:
		pragma platform linux:
			void main() {
				if (PLATFORM.lower() == "linux")
					systemd_notify("WATCHDOG=1"); // Only useful within a systemd service.
				else
					alert("Info", "This example only works on Linux");
			}



Advanced Topics for C++ Developers:
	This section contains information that is useful for anyone who wishes to interact in any way with NVGT's source code. Whether someone wants to perform a simple sourcecode build, develop an NVGT plugin dll, learn how to contribute to the engine or just learn about how NVGT works internally, you'll find such information here. The topics that will be discussed here will generally not be applicable for those who just want to develop games using NVGT using the installer downloaded from nvgt.dev.


	Building NVGT for Android:
		To build NVGT for Android, you will need the Android SDK. The easiest way to get this at least on Windows is if you have Visual Studio to download the Mobile Development workload, or else to install Android Studio.

		As with other platforms, most of the work regarding building libraries has been done for you, you can download [droidev.zip](https://nvgt.dev/droidev.zip) here, and extract it similarly to how it is described above for other platforms. Look at jni/example-custom.mk to learn how to set a custom location to this directory.

		On some systems, the Android SDK requires a bit of setup to use it extensively enough to build NVGT. If you are using Android Studio directly or if anything else has configured your Android SDK for you it is unlikely you'll need any of this, but in case you need to set up your SDK manually especially on Windows, here are a few notes:
		1. The Android SDK on Windows must not be in a directory containing spaces, or at least must not be invoked with one. For example, the Mobile Development workload on Visual Studio by default installs the tools in "C:\Program Files (x86)\Android". So to use it, you may need to symlink the folder with a command run as administrator like `mklink /d C:\android "C:\Program Files (x86)\Android"`.
		2. It may also be a good idea to run the Gradle tool (described below) as administrator if you get any weird access denied errors when trying to do it otherwise, you'll see it's trying to write to some file within the Android SDK directory which may require administrative rights to access depending on where it is installed. If you keep getting errors even after doing this, run `gradlew -stop` to kill any old Gradle daemons that were running without privileges.
		3. There are a couple of environment variables you may need to set before building. These are the `ANDROID_NDK_HOME`, `ANDROID_HOME`, and `JAVA_HOME` variables which help locate the Android development tools for anything that needs them. An example of setting them at least for Windows is in other/droidev.bat.
		4. If at any point you need to run sdkmanager to accept licenses on Windows, run it... yep, as admin. Otherwise you will be informed that the licenses have been successfully accepted, but when you try using the tools, you'll be told to accept them again.
		5. It should be noted that both Visual Studio and Android Studio install a functioning Java runtime environment for you, so you don't have to go hunting for one. On Visual Studio, this may be at "C:\Program Files (x86)\Android\openjdk\jdk-17.0.8.101-hotspot", and for Android Studio it may be at "C:\Android\Android Studio\jre". Sadly these paths might change a bit over time (especially the JDK version in the Visual Studio path), so you may need to look up the latest paths if you are having trouble.

		Once the Android SDK is set up, cd to the jni directory and run `gradlew assembleRunnerDebug assembleStubRelease`

		On platforms other than Windows you may need to run `./gradlew` instead of `gradlew`.

		If you want to install the runner application to any devices connected in debug mode, you can run `gradlew installRunnerDebug`.

		Do not directly install the stub APK on your device, only do that for the runner application. In regards to the stub, after the Gradle build command succeeds, there is now a file in release/stub/nvgt_android.bin which should be moved into the stub folder of your actual NVGT installation, allowing NVGT to compile .APK files from game source code.



	Building NVGT for linux:

	Building with the `build_linux.sh` script:
		There is a [script to build NVGT on Linux](https://raw.githubusercontent.com/samtupy/nvgt/main/build/build_linux.sh) (tested on Debian and Ubuntu). It tends to build pretty portably so you can run it basically anywhere, and it will attempt to successfully download all required dependencies and build them for you. The result will be a fully built NVGT.

		Internally, this script is used within our GitHub Actions to make builds of NVGT. It is also used within our local testing environments.

	Notes:
		* This script will currently only run on systems where `apt` and `pip` are installed, and does not support any other package managers.
		* This script will create and activate a [virtual environment](https://docs.python.org/3/library/venv.html).

		This script can be ran in two modes:
		* Adding `ci` as an argument causes the dependencies to be downloaded in the current working directory inside a `deps` folder (useful if you already are working from within NVGT's source directory).
		* If `ci` is not present, the script will assume NVGT is not downloaded and will clone NVGT into the current directory before attempting to build it.

	Example of Running the script with the `ci` argument:
		It is assumed you are in a freshly-cloned NVGT, so that your working directory ends with `nvgt`.
		chmod +x build/build_linux.sh
		./build/build_linux.sh ci

		It will then attempt to download all required packages and build NVGT. This will take some time.

	Example of Running the script without the `ci` argument:
		Insure you are in a working directory where you are okay with the script making a few folders; in particular `deps` and `nvgt`. This is where all of the downloading, building, etc. will occur. The below example assumes that build_linux.sh is in the same directory, but it does not assume NVGT is already downloaded.
		chmod +x build_linux.sh
		./build_linux.sh

	Building NVGT manually:
		If you wish to build manually, some rather old instructions are below. At this time, it would probably be much more beneficial to read the `build_linux.sh` script or [readme.md](https://github.com/samtupy/nvgt) for much more updated commands; the below commands are here for reference and aren't updated often.

		Please keep in mind that this is a very very rough draft, I've only done this once before when I built nvgt's server components for stw. This will attempt to describe, even for a user who doesn't use linux much, how to build nvgt at least on Ubuntu 22.04 LTS.

	tools:
		You will need the GNU compiler collection / GNU make / a few other tools. You can see if you already have these on your installation by running `gcc`, `g++`, `make`. If this fails, run `sudo apt install build-essential gcc g++ make autoconf libtool`.

	commands:
		```bash
		mkdir deps && cd deps
		git clone https://github.com/codecat/angelscript-mirror
		cd deps/angelscript-mirror/sdk/angelscript/projects/gnuc
		make
		sudo make install

		sudo apt install libssl-dev libcurl4-openssl-dev libopus-dev libsdl2-dev
		sudo apt remove libsdl2-dev

	Note:
		The first command installs a version of SDL that is too old, but still installs loads of deps. Now we will build sdl.

		`cd deps`

		Before continuing, download sdl into a folder called SDL.

		```bash
		mkdir SDL_Build
		cd SDL_Build
		cmake ../SDL
		cmake --build .
		sudo cmake --install .

		cd deps
		git clone https://github.com/pocoproject/poco
		cd poco
		./configure --static --no-tests --no-samples --cflags=-fPIC
		make -s -j2

	Note:
		The 2 in `make -j2` is how many CPU cores you would like to use when building. Change this to the number of CPU cores you would like to use. If you do not know how many cores your system has, you can use the `lscpu` command on many distributions to check.

		```bash
		sudo make install

		cd deps
		git clone https://github.com/lsalzman/enet
		cd enet
		autoreconf -vfi
		./configure
		make
		sudo make install

		cd deps
		git clone https://github.com/bulletphysics/bullet3
		cd bullet3
		./build_cmake_pybullet_double.sh
		cd cmake_build
		sudo cmake --install .

		```bash
		cd deps
		git clone https://github.com/libgit2/libgit2
		cd libgit2
		mkdir build
		cd build
		cmake ..
		cmake --build .
		sudo cmake --install .

		You will need scons, which you can get by running pip3 install scons.

	Finally...:
		cd to the root of the nvgt repository and extract https://nvgt.dev/lindev.tar.gz to a lindev folder there.

		scons -s

		Enjoy!


	Building NVGT for MacOS:

	Building with the `build_macos.sh` script:
		There is a [script to build NVGT on macOS](https://raw.githubusercontent.com/samtupy/nvgt/main/build/build_macos.sh). It will build pretty portably so you can run it basically anywhere (assuming you have Homebrew and the Xcode command line tools). It will attempt to successfully download all required dependencies and build them for you. The result will be a fully built NVGT.

		Internally, this script is used within our GitHub Actions to make builds of NVGT. It is also used within our local testing environments.

	Notes:
		* This script will create and activate a [virtual environment](https://docs.python.org/3/library/venv.html).

		This script can be ran in two modes:
		* Adding `ci` as an argument causes the dependencies to be downloaded in the current working directory inside a `deps` folder (useful if you already are working from within NVGT's source directory).
		* If `ci` is not present, the script will assume NVGT is not downloaded and will clone NVGT into the current directory before attempting to build it.

	Example of Running the script with the `ci` argument:
		It is assumed you are in a freshly-cloned NVGT, so that your working directory ends with `nvgt`.
		chmod +x build/build_macos.sh
		./build/build_macos.sh ci

		It will then attempt to download all required packages and build NVGT. This will take some time.

	Example of Running the script without the `ci` argument:
		Insure you are in a working directory where you are okay with the script making a few folders; in particular `deps` and `nvgt`. This is where all of the downloading, building, etc. will occur. The below example assumes that build_macos.sh is in the same directory, but it does not assume NVGT is already downloaded.

		chmod +x build_macos.sh
		./build_macos.sh


	Building NVGT manually:
		If you wish to build manually, some rather old instructions are below. At this time, it would probably be much more beneficial to read the `build_macos.sh` script or [readme.md](https://github.com/samtupy/nvgt) for much more updated commands; the below commands are here for reference and aren't updated often.

		Assuming xcode and homebrew are installed:

		```bash
		pip3 install scons
		brew install autoconf automake libgit2 libtool openssl sdl2 bullet

		mkdir deps
		git clone https://github.com/codecat/angelscript-mirror
		cd "angelscript-mirror/sdk/angelscript/projects/cmake"
		mkdir build; cd build
		cmake ..
		cmake --build .
		sudo cmake --install .

		cd deps
		git clone https://github.com/lsalzman/enet
		cd enet
		autoreconf -vfi
		./configure
		make
		sudo make install

		cd deps
		git clone https://github.com/pocoproject/poco
		cd poco
		./configure --static --no-tests --no-samples
		make -s -j<n> where <n> is number of CPU cores to use
		sudo make install

		cd deps
		git clone https://github.com/libgit2/libgit2
		cd libgit2
		mkdir build
		cd build
		cmake ..
		cmake --build .
		sudo cmake --install .

		cd nvgt
		scons -s


	Debugging Scripts:
		One very useful feature of NVGT is the ability to debug the scripts that you write. This means being able to pause script execution at any time (either triggered manually or automatically), view what's going on and even make changes or inject code, and then resume the execution. You can even execute one statement in your code at a time to get an idea of exactly what it is doing.

	The -d or --debug option:
		To debug a script in nvgt, it is required that you use the command line version of nvgt (nvgt.exe). On platforms other than Windows, only a command line version is available.

		There is a command line argument you must pass to nvgt.exe along with the script you want to run in order to debug it, which is either -d, or --debug depending on what you prefer.

		So for example, open a command prompt or terminal and change directory to the place where you have stored your game. You could then run, for example, `nvgt -d mygame.nvgt` assuming a correct installation of nvgt, which would cause mygame.nvgt to be launched with the Angelscript debugger getting initialized.

	the debugging interpreter:
		When you run a script with the debugger, it will not start immediately. Instead, you will be informed that debugging is in progress, that the system is waiting for commands, and that you can type h for help.

		If the last line on your terminal is \[dbg\]\> , you can be sure that the system is waiting for a debug command.

		If you press enter here without first typing a command, the last major debugger action is repeated. This is not necessarily the last command you have typed, but is instead the last major action (continue, step into, step over, step out). The default action is to continue, meaning that unless you have changed the debugger  action, pressing enter without typing a command will simply cause the execution of your script to either begin or resume where it left off.

		Pressing ctrl+c while the debug interpreter is open will exit out of nvgt completely similar to how it works if the debugger is not attached. Pressing this keystroke while the interpreter is not active will perform a user break into the debugger, meaning that your script will immediately stop executing and the debug interpreter will appear.

		To list all available commands, type h and enter. We won't talk about all of the commands here, but will discuss a few useful ones.

	Useful debugging commands:
		* c: set debugger action to continue, execute until next breakpoint or manual break
		* s: set debugger action to step into, only execute the next instruction
		* n: set debugger action to step over, execute until next instruction in current function
		* o: set debugger action to step out, execute until the current function returns
		* e \<return-type\> \<expression\>: Evaluate a simple code statement, anything from a simple math expression to calling a function in your script.
		* p \<value\>: print the value of any existing variable
		* l: list various information, for example "l v" would list local variables
		* b: set a function or file breakpoint, type b standalone for instructions
		* a: abort execution (more graceful version of ctrl+c)

	registered debugging functions:
		If you are working with a troublesome bit of code, you may find that breaking into the debugger at a specific point is a very helpful thing to do. For this situation, NVGT has the `debug_break()` function.

		This function will do nothing if called without a debugger attached or from a compiled binary, so it is OK to leave some calls to this around in your application so long as you are aware of them for when you run your script with the debugger. This function will immediately halt the execution of your script and cause the debugger interpreter to appear.

		You can also programmatically add file and function breakpoints with the functions `void debug_add_file_breakpoint(string filename, int line_number)` and `void debug_add_func_breakpoint(string function_name)`.

	breakpoints:
		To describe breakpoints, we'll break (pun intended) the word into it's 2 parts and describe what the words mean in this context.
		* When debugging, a break means pausing your script's execution and running the debugging interpreter.
		* In this context, a point is either a file/line number combo or a function name which, if reached, will cause a debugger break.

		For example if you type the debug command "b mygame.nvgt:31" and continue execution, the debugging interpreter will run and the script execution will halt before line31 of mygame.nvgt executes.

		It is also possible to break into the debugger whenever a function is about to execute, simply by passing a function name instead of a file:line combo to the b debug command.

	notes:
		* It is worth noting that when the debugger interpreter is active, your script's execution is completely blocked. This means that any window you have shown will not be processing messages. Some screen readers don't like unresponsive windows that well, so be careful when breaking into the debugger while your game window is showing! Maybe in the future we can consider a setting that hides the game window whenever a debug break takes place.
		* This debugger has not been tested very well in a multi-threaded context, for example we do not know what happens at this time if 2 threads call the debug_break() function at the same time. We intend to investigate this, but for now it's best to debug on the main thread of your application. In particular no commands exist as of yet to give contextual thread information.


	Notes on NVGT's documentation generator:

	format of the src directory:

		* topics may be in .md or .nvgt format at this time. When NVGT becomes a c++ library as well, we may add ability to perform basic parsing of .h files.
		* Directories are subsections and can be nested.
		* Sections and subsections, starting at NVGTRepoRoot/doc/src, are scanned for topics with the following rules in mind:
		    * If a .index.json file exists, this file is loaded and parsed to create the topic list. The json is a list containing either strings or 2-element lists. If a string, just indicates a topic filename. If a list, the first element is a topic filename string and the second is a topic display name string if the topic filename isn't adequate.
		    * If a .index.json file does not exist, topics/subsections are loaded in alphabetical order. A topic C will appear before a subsection F.
		    * Unless a .index.json file specifies a topic display name, the topic filename minus the extension is used as the topic name. Punctuation characters at the beginning of the display name are removed to provide for more flexible sorting. A subsection !C will show before a subsection F, but !C will display C thus allowing for multiple sorting levels without a .index.json file.
		    * If there is a .MDRoot file present in a directory, the contents of that directory are output in new markdown/html documents instead of being appended to the main/parent documents, and a link to the new document is appended to the parent one.
		    * If a topic filename contains an @ character right before the extension, that document is treated as a new markdown root document in the final output instead of being appended to the main/parent document, this is just a way to invoque the same as above but for a single file.
		    * If a topic filename contains a + before the extension, the first line in the file minus markdown heading indicators will be used as the topic display name instead of a titlecased topic filename.
		* .nvgt files contain embedded markdown within comments. Most markdown syntax is the same, except that nvgt's docgen script replaces single line breaks with double lines so that each line is a markdown paragraph. Start a docgen comment with /**, and start a docgen comment with the above mentioned linefeed behavior disabled with /**\.
		* Another embedded markdown comment within .nvgt files is the "// example:" comment, which will be replaced with "## example" in markdown.
		* When parsing a .nvgt file, a "# topicname" markdown directive is added to the top of the output to avoid this redundant step in example functions.
		* The docgen program creates a .chm file which requires parsing this markdown into html, and there may be reasons for removing embedded markdown indentation anyway. This resulted in an indentation rule where tabs are stripped from the document when passed to the python markdown package, spaces are not and can be used for things like nested lists.
		* If the very first topic in any category begins with a heading with the same name as the containing category, the heading name is stripped from the markdown and html output of the documentation, and the heading indentation of that topic is set to that of the parent category. This allows one to easily create intro sections for categories without creating duplicate headings with the same name. The heading is stripped after the single html/chm version of that topic is printed, as such a heading should remain in the chm documentation.

	Installing the Microsoft HTML help compiler:
		Because of it's simple format and easy distribution, we still prefer to generate the NVGT documentation as a .chm file (compressed HTML help). Unfortunately, the link to the html help workshop installer has been broken by Microsoft for a couple of years now. Fortunately, the installer for this program was archived from Microsoft's official website by the wayback machine. Until we get a better link, you should be able to [download it here](http://web.archive.org/web/20200312222543/http://download.microsoft.com/download/0/A/9/0A939EF6-E31C-430F-A3DF-DFAE7960D564/htmlhelp.exe), though it should be noted that only those wishing to rebuild nvgt's documentation from source will need this program.


	Plugin Creation:
		Does NVGT not provide the function you need, and do you know a bit of c++? If so, perhaps NVGT plugins are exactly what you're looking for!

		This document will describe all there is to know about creating nvgt plugins, both dynamically and statically.

	What is a plugin in the context of NVGT?:
		An NVGT plugin, in it's most basic form, is simply a module of code that is executed during the script loading part of the engine initialization process, one which can extend the functionality of NVGT by directly gaining access to and registering functions with it's internal Angelscript engine.

		Plugins are not just limited to functions, but classes, enums, funcdefs and anything else one could register normally using the Angelscript scripting library.

	Types of plugin builds:
		A plugin can either be a shared library (.dll / .dylib / .so) that gets loaded when needed, or a static library (.lib / .a) that is linked directly into a custom build of NVGT. Both methods have different advantages and disadvantages.

		Dynamically loaded plugins, those built into a shared library, are easier to get working with NVGT because it's far easier to create such a plugin without at all altering NVGT's build process or adding things to it. You could use your own build system and your own environment, so long as the proper ABI is exposed to NVGT in the end and an up-to-date version of Angelscript is used within your plugin. However, a smart player may figure out how to replace your plugin dll with some sort of malicious copy, your dll plugin could be duplicated and reused in other projects, you'll have an extra dll file to release with your game distribution etc.

		Static plugins on the other hand, while a bit tougher to build, are certainly more rewarding in the end. From plugin code being packaged directly into your binary to a smaller distribution size because of no duplicated crt/other code in a dll to direct integration with NVGT's build system, there are several advantages that can be observed when choosing to create a static plugin.

		If one chooses to follow every step of the existing NVGT plugin creation process that is used internally by engine developers, you can set up your plugin such that it can easily be built either dynamically or statically depending on the end-user's preference.

	The basic idea:
		In short, the idea here stems from a pretty simple base. The user creates a .cpp file that includes the nvgt_plugin.h header that can do any magic heavy lifting needed, then the user just defines an entry point using a macro declared in nvgt_plugin.h. This entry point receives a pointer to the asIScriptEngine instance used by NVGT, which the plugin developer can do anything they please with from registering custom functions to installing some sort of custom profiler. The entry point can return true or false to indicate to NVGT whether the plugin was able to successfully initialize.

		This plugin entry point always takes one argument, which is a structure of data passed to it by NVGT. The structure contains the Angelscript engine pointer as well as pointers to several other Angelscript functions that may be useful, and may be expanded with pointers to other useful interfaces from NVGT as well. One just simply needs to call a function provided by nvgt_plugin.h called prepare_plugin passing to it a pointer to the aforementioned structure before their own plugin initialization code begins to execute.

		To link a static plugin with the engine assuming the nvgt's build script knows about the static library file, one need only add a line such as static_plugin(\<plugname\>) to the nvgt_config.h file where \<plugname\> should be replaced with the name of your plugin.

	small example plugin:
	include <windows.h>:
	include "../../src/nvgt_plugin.h":

		void do_test() {
			MessageBoxA(0, "It works, this function is being called from within the context of an NVGT plugin!", "success", 0);
		}

		plugin_main(nvgt_plugin_shared* shared) {
			prepare_plugin(shared);
			shared->script_engine->RegisterGlobalFunction("void do_test()", asFUNCTION(do_test), asCALL_CDECL);
			return true;
		}

	picking it apart:
		We shall forgo any general comments or teaching about the c++ language itself here, but instead will just focus on the bits of code that specifically involve the plugin interface.

		The first thing that you probably noticed was this include directive which includes "../../src/nvgt_plugin.h". Why there? While this will be described later, the gist is that NVGT's build setup already has some infrastructure set up to build plugins. NVGT's github repository has a plugin folder, and in there are folders for each plugin. This example is using such a structure. We will talk more in detail about this later, but for now it is enough to know that nvgt_plugin.h does not include anything else in nvgt's source tree, and can be safely copy pasted where ever you feel is best for your particular project (though we do recommend building plugins with NVGT's workspace).

		The next oddity here, why doesn't the plugin_main function declaration include a return type? This is because it is a macro defined in nvgt_plugin.h. It is required because the name of the entry point will internally change based on whether you are compiling your plugin statically or dynamically. If you are building your plugin as a shared library, the function that ends up exporting is called nvgt_plugin. However since one of course cannot link 2 static libraries with the same symbol names in each to a final executable, the entry point for a static plugin ends up being called nvgt_plugin_\<plugname\> where \<plugname\> is replaced with the value of the NVGT_PLUGIN_STATIC preprocessor define (set at plugin build time). In the future even dynamic libraries may possibly contain the plugin name in their entry point function signatures such that more than one plugin could be loaded from one dll file, but for now we instead recommend simply registering functions from multiple plugins in one common entry point if you really want to do that.

		Finally, remember to call prepare_plugin(shared) as the first thing in your plugin, and note that if your entry point does not return true, this indicates an error condition and your plugin is not loaded.

	NVGT's plugin building infrastructure:
		As mentioned a couple of times above, NVGT's official repository already contains the infrastructure required to build plugins and integrate them with NVGT's existing build system, complete with the ability to exclude some of your more private plugins from being picked up by the repository. While it is not required that one use this setup and in fact one may not want to if they have a better workspace set up for themselves, we certainly recommend it especially if you are making a plugin that you may want to share with the NVGT community.

	The plugin directory:
		In nvgt's main repository, the plugin directory contains all publicly available plugins. Either if you have downloaded NVGT's repository outside of version control (such as a public release artifact) or if you intend to contribute your plugin to the community by submitting a pull request, you can feel free to use this directory as well.

		Here, each directory is typically one plugin. It is not required that this be the case, other directories that are not plugins can also exist here, however any directory within the plugin folder that contains a file called _SConscript will automatically be considered as a plugin by the SConstruct script that builds NVGT.

		The source code in these plugins can be arranged any way you like, as it is the _SConscript file you provide that instructs the system how to build your plugin.

		An example _SConscript file for such a plugin might look like this:
	Import the SCons environment we are using:
		Import("env")

	Create the shared version of our plugin if the user has not disabled this feature.:
		if ARGUMENTS.get("no_shared_plugins", "0") == "0":
			env.SharedLibrary("#release/lib/test_plugin", ["test.cpp"], libs = ["user32"])

	If we want to make a static version along side our shared one, we need to specifically rebuild the object file containing the plugin's entry point with a different name so that SCons can maintain a proper dependency graph. Note the NVGT_PLUGIN_STATIC define.:
		static = env.Object("test_plugin_static", "test.cpp", CPPDEFINES = [("NVGT_PLUGIN_STATIC", "test_plugin")])
	now actually build the static library, reusing the same variable from above for fewer declarations.:
		static = env.StaticLibrary("#build/lib/test_plugin", [static])

	Tell NVGT's SConstruct script that the static version of this plugin needs symbols from the user32 library.:
		static = [static, "user32"]

	What is being returned to NVGT's SConstruct in the end is a list of additional static library objects that should be linked.:
		Return("static")

		Note that while the above example returns the user32 library back to NVGT's build script, it should be noted that most system libraries are already linked into nvgt's builds. The example exists to show how an extra static library would be passed to NVGT from a plugin if required, but this should only be done either as a reaction to a linker error or if you know for sure that your plugin requires a dependency that is not automatically linked to NVGT, examples in the git2, curl or sqlite3 plugins.

	the user directory:
		NVGT's github repository also contains another root folder called user. This is a private scratchpad directory that exists so that a user can add plugins or any other code to NVGT that they do not want included in the repository.

		First, the repository's .gitignore file ignores everything in here accept for readme.md, meaning that you can do anything you like here with the peace of mind that you won't accidentally commit your private encryption plugin to the public repository when you try contributing a bugfix to the engine.

		Second, if a _SConscript file is present in this directory, NVGT's main build script will execute it, providing 2 environments to it via SCons exports. The nvgt_env environment is what is used to directly build NVGT, for example if you need any extra static libraries linked to nvgt.exe or the stubs, you'd add one by importing the nvgt_env variable and appending the library you want to link with to the environment's LIBS construction variable.

		Last but not least, if a file called nvgt_config.h is present in the user folder, this will also be loaded in place of the nvgt_config.h in the repo's src directory.

		You can do whatever you want within this user directory, choosing to either follow or ignore any conventions you wish. Below is an example of a working setup that employs the user directory, but keep in mind that you can set up your user directory any way you wish and don't necessarily have to follow the example exactly.

	user directory example:
		The following setup is used for Survive the Wild development. That game requires a couple of proprietary plugins to work, such as a private encryption layer.

		In this case, what was set up was a second github repository that exists within the user directory. It's not a good idea to make a github repository out of the root user folder itself because git will not appreciate this, but instead a folder should be created within the user directory that could contain a subrepository. We'll call it nvgt_user.

		The first step is to create some jumper scripts that allow the user folder to know about the nvgt_user repository contained inside it.

		user/nvgt_config.h:
	include "nvgt_user/nvgt_config.h":
		and

		user/_SConscript:
		Import(["plugin_env", "nvgt_env"])
		SConscript("nvgt_user/_SConscript", exports=["plugin_env", "nvgt_env"])

		Now, user/nvgt_user/nvgt_config.h and user/nvgt_user/_SConscript will be loaded as they should be, respectively.

		In the nvgt_user folder itself we have _SConscript, nvgt_plugin.h, and some folders containing private plugins as well as an unimportant folder called setup we'll describe near the end of the example.

		nvgt_config.h contains the custom encryption routines / static plugin configuration that is used to build the version of NVGT used for Survive the Wild.

		The user/nvgt_user/_SConscript file looks something like this:
		Import("plugin_env", "nvgt_env")

		SConscript("plugname1/_SConscript", variant_dir = "#build/obj_plugin/plugname1", duplicate = 0, exports = ["plugin_env", "nvgt_env"])
		SConscript("plugname2/_SConscript", variant_dir = "#build/obj_plugin/plugname2", duplicate = 0, exports = ["plugin_env", "nvgt_env"])
	nvgt_user/nvgt_config.h statically links with the git2 plugin, lets delay load that dll on windows so that users won't get errors if it's not found.:
		if nvgt_env["PLATFORM"] == "win32":
			nvgt_env.Append(LINKFLAGS = ["/delayload:git2.dll"])
		And finally an _SConscript file for nvgt_user/plugname\* might look something like this:
		Import(["plugin_env", "nvgt_env"])

		static = plugin_env.StaticLibrary("#build/lib/plugname2", ["code.cpp"], CPPDEFINES = [("NVGT_PLUGIN_STATIC", "plugname2")], LIBS = ["somelib"])
		nvgt_env.Append(LIBS = [static, "somelib"])
		As you can see, the decision regarding the custom plugins used for Survive the Wild is to simply not support building them as shared libraries, as that will never be needed from the context of that game.

		The only other item in the private nvgt_user repository used for Survive the Wild is a folder called setup, and it's nothing but a tiny all be it useful convenience mechanism. The setup folder simply contains copies of the user/_SConscript and user/nvgt_config.h files that were described at the beginning of this example, meaning that if nvgt's repository ever needs to be cloned from scratch to continue STW development (such as on a new workstation), the following commands can be executed without worrying about creating the extra files that are outside of the nvgt_user repository in the root of the user folder:

		```bash
		git clone https://github.com/samtupy/nvgt
		cd nvgt/user
		git clone https://github.com/samtupy/nvgt_user
		cp nvgt_user/setup/* .

		And with that, nvgt is ready for the private development of STW all with the custom plugins still being safely in version control! So long as the cwd is outside of the nvgt_user directory the root nvgt repository is effected, and once inside the nvgt_user directory, that much smaller repository is the only thing that will be touched by git commands.

		Remember, you should use this example as a possible idea as to how you could potentially make use of NVGT's user directory, not as a guide you must follow exactly. Feel free to create your own entirely different workspace in here or if you want, forgo use of the user directory entirely.

	Angelscript addon shims:
		When creating a shared/dynamic plugin made up of more than 1 cpp file, you must `#define NVGT_PLUGIN_INCLUDE` and then `#include "nvgt_plugin.h"` before you `#include <angelscript.h>`. This is to make sure that any additional units you build will use the manually imported Angelscript symbols that were passed to the plugin from NVGT, a multiple defined symbol error might appear otherwise or if your code compiles, 2 different versions of the Angelscript functions could be used which would be very unsafe. To make it easier to include Angelscript addons into your plugins, little shims are provided for the common addons in the nvgt repository in the ASAddon/plugin directory. You can simply compile ASAddon/plugin/scriptarray.cpp, for example, and the scriptarray plugin will safely be included into your plugin. nvgt_plugin.h should still be included before scriptarray.h in any of your plugin source files, however.

		When compiling a static plugin, you do not need to bother linking with these addon shims, because in that case your plugin's static library will be linked with NVGT when NVGT is next recompiled, and NVGT already contains working addons.

	plugin dll loading:
		A common sanario is that you may wish to make a plugin that then loads another shared library. This happens already in nvgt, particularly with the git2 plugin. The plugin itself is called git2nvgt, but it loads git2.dll. This calls for some consideration.

		NVGT applications generally put their libraries in a lib folder to reduce clutter in the applications main directory. This isn't required, and in fact if you want to move all shared objects out of the lib folder and put them along side your game executable, you actually tend to avoid the small issue mentioned here. The problem is that sometimes, we need to explicitly tell the operating system about the lib directory at runtime (right now true on windows) in order for it to find the shared libraries there.

		For shared/dynamic plugins, this isn't really an issue. NVGT has launched, informed the system of the lib directory, and loaded your plugin in that order meaning that any dll your plugin imports can already be located and imported just fine.

		With static plugins, on the other hand, we must work around the fact that the operating system will generally try loading all libraries that the program uses before any of that program's code executes, and will show the user an error message and abort the program if it can't find one. Returning to the git2 example from earlier, if you were to ship this plugin with your game, a file would exist called lib/git2.dll on windows. When the static library git2nvgt.lib is created, it will add git2.dll to it's import table but with no knowledge of the lib folder at that point. When the custom build of NVGT which includes this plugin tries to run, an error message will appear because git2.dll can't be found, on account of NVGT never being able to tell the system about the lib directory before the operating system evaluates nvgt's import table and tries to load the libraries found within.

		The solution, at least on windows, is delayed dll loading. It is demonstrated above in the user directory example, but it's easy to gloss over considering it's level of importants. If you add the linkflag `/delayload:libname.dll` to your static plugin's build script, now NVGT's code is allowed to execute meaning it can tell the system about the lib directory, and then the dll will load the first time a function is called from it. On MacOS/clang, there is the linkflag -weak_library which does something similar, used like `-weak_library /path/to/library.dylib`.

		Delay loading does has the disadvantage that the app tends to crash if the dll is not present when it is needed rather than giving the user a nice error message, but you can work around that by manually loading the dll with the LoadLibrary function on windows / similar facilities on other platforms then immediately unloading it just to see if the system will be able to find it, and you can choose to show an error message in that case if you wish.

	cross platform considerations:
		If you are only building plugins for projects that are intended to run on one platform, this section may be safely skipped. However if your game runs on multiple platforms and if you intend to introduce custom plugins, you probably don't want to miss this.

		There are a couple of things that should be considered when creating plugins intended to run on all platforms, but only one really big one. In short, it is important that a cross platform plugin's registered Angelscript interface looks exactly the same on all platforms, even if your plugin doesn't support some functionality on one platform. For example if your plugin has functions foo and bar but the bar function only works on windows, it is important to register an empty bar function for any non-windows builds of your plugin rather than excluding the bar function from the angelscript registration of such a plugin build entirely. This is especially true if you intend to, for example, cross compile your application with the windows version of NVGT to run on a linux platform.

		The reasoning is that Angelscript may sometimes store indexes or offsets to internal functions or engine registrations in compiled bytecode rather than the names of them. This makes sense and allows for much smaller/faster compiled programs, but what it does mean is that NVGT's registered interface must appear exactly the same both when compiling and when running a script. Maybe your plugin with foo and bar functions get registered into the engine as functions 500 and 501, then maybe the user loads a plugin after that with boo and bas functions that get registered as functions 502 and 503. Say the user makes a call to the bas function at index 503. Well, if the foo bar plugin doesn't include a bar function on linux builds of it, now we can compile the script on windows and observe that the function call to bas at index 503 is successful. But if I run that compiled code on linux, since the bar function is not registered (as it only works on windows), the bas function is now at index 502 instead of 503 where the bytecode is instructing the program to call a function. Oh no, program panic, invalid bytecode! The solution is to instead register an empty version of the bar function on non-windows builds of such a plugin that does nothing.

	Angelscript registration:
		Hopefully this document has helped you gather the knowledge required to start making some great plugins! The last pressing question we'll end with is "how does one register things with NVGT's Angelscript engine?" The angelscript engine is a variable in the nvgt_plugin_shared structure passed to your plugins entry point, it's called script_engine.

		The best reference for how to register things with Angelscript is the Angelscript documentation itself, and as such, the following are just a couple of useful links from there which should help get you on the right track:
		* [registering the application interface](https://www.angelcode.com/angelscript/sdk/docs/manual/doc_register_api_topic.html)
		* [registering a function](https://www.angelcode.com/angelscript/sdk/docs/manual/doc_register_func.html)
		* [registering global properties](https://www.angelcode.com/angelscript/sdk/docs/manual/doc_register_prop.html)
		* [registering an object type](https://www.angelcode.com/angelscript/sdk/docs/manual/doc_register_type.html)

		Good luck creating NVGT plugins, and feel free to share some of them to the community if you deem them worthy!


appendix:
	This section contains miscellaneous documents such as license agreements, lists of static information, etc.


	To do list:
		NVGT is still a relatively young project with many areas that need improvement, and there are also several long-term goals for this project that are not yet realized. This document attempts to keep track of any intended development activity that has not yet taken place, both for existing NVGT developers and for anyone looking to find a way to help contribute to the engine.

	User facing items:
		Items in this section generally effect end-users in some way and are not related directly to the codebase. Items may include classes we intend to add to the engine or areas of the website we wish to improve.

	Documentation:
		There has been absolutely loads of work so far put into documenting NVGT's API reference and feature set, but the task is not complete. Any contributions to the docs are welcome.

	examples:
		We have been holding off on these until the API stabalizes so that we don't teach people things that are soon to be changed and because we're limited to a public domain sound catelog, but we need to create several example projects that can help get people started with NVGT. Not just for the docs, but full runnable projects in the other repository. They should range from basic card game to small platformer to online client where people can walk around.

	Joystick object:
		The joystick class in NVGT is still a no-op interface. There has already been a bit of code to begin the process of wrapping SDL's joystick support, but it is not yet complete.

	tone_synth:
		Another object from BGT we have not yet reimplemented, though the beginnings of tone generation will appear in miniaudio.

	Speech dispatcher and the `tts_voice` object:
		Currently, NVGT's Speech Dispatcher implementation for Linux only works with the screen reader speech functions. At this time, we are still considering if we should implement it into the `tts_voice` object as well.

	VSCode extension:
		A plan that has existed for a few months now is to create a VSCode extension for Angelscript that works with NVGT scripts. To facilitate this we have wrapped a function called script_dump_engine_configuration, an example of which you can see in test/quick/dump_engine_config.nvgt. This function dumps a complete reference of everything registered in the engine, enough to compile scripts. This will, once time permits to learn the needed components, allow us to create an extension for VSCode that allows everything from symbol lookup to intellisense. As of 0.89.0 many of the function and method parameters have been properly named, which helps us get just a bit closer to this goal.

	JAWS keyhook:
		There has been loads of progress made with NVGT's JAWS keyhook, and it should now work in almost all senarios. The only thing to be aware of is that if JAWS crashes, you may have to alt+tab a couple of times. Other than that though, the keyhook is stable and useable!

	SDL dialog boxes:
		At the moment, we are using SDL's message box system to show simple dialogs rather than implementing it on our own. However, this implementation is not ideal for 3 reasons.
		1. shortcuts with an ampersand don't work, we can't create a button called `&yes` that works with alt+y.
		2. Copying text does not work with ctrl+c.
		3. No internationalization, yes and no buttons are English on non-English windows.
		Either we will see if SDL will improve message boxes soon, or switch to something else.

	Switch to miniaudio (in progress):
		Currently we use the Bass audio library for sound output, which functionally speaking does work great. However Bass is not open source, and a commercial license must be purchased from [Un4seen](https://www.un4seen.com/bass.html) in order to sell commercial projects. For NVGT, this is not ideal and Bass was only used because it worked quite well at the time that NVGT was only being used to bolster Survive the Wild development with no opensource intentions. Instead, we plan to switch to [miniaudio](https://github.com/mackron/miniaudio) which is open source and in the public domain, and thus which will solve such commercial licensing issues. There has been much progress here and we are hopefully to have miniaudio integrated soon.

	Recording from a microphone:
		Especially since Survive the Wild has implemented voice chat support, people rightfully wonder how to record audio in NVGT. Survive the Wild does this with a plugin specifically designed for it's voice chat. The API is not one which we wish to support publicly as it is very limited and confined to stw's use case. This should come soon as miniaudio implementation has made much progress.

	Subscripting improvements:
		* NVGT allows a scripter to execute Angelscript code from within their Angelscript code, such as the python eval function. The user is given control of what builtin NVGT functions and classes these subscripts have access to, but it's still a bit rough. Basically Angelscript provides us with this 32 bit DWORD where we can map certain registered functions to bitflags and restrict access to them if a calling module's access bitmask doesn't include a flag the functions were registered with. However this means that we have 32 systems or switches to choose from, so either we need to assign builtin systems to them in a better way, or investigate this feature Angelscript has which is known as config groups and see if we can use them for permission control. C++ plugins in particular complicate this issue.
		* Register asIScriptObject and asITypeInfo which would allow people to traverse script classes and execute functions in them.

	Provide user facing pack file encryption:
		Currently pack file encryption uses internal methods requiring a user to rebuild NVGT to change the encryption routines, but there is full intention of also adding a `pack.set_encryption` so that users of NVGT can also manage how their packs are encrypted. A secondary SQL based pack file object has been contributed to the project which does include this functionality though we still intend to make it a part of the smaller, primary pack object. A contributor is in the process of implementing this functionality to be included with miniaudio.

	CI improvements:
		NVGT is now set up to work with the Github releases page for it's next release, and at this time anyone can fork the project and run the release action to build the latest development version of the engine. However, the process isn't yet automated and we'd like to set up builds on some sort of schedule so that users can download the latest development NVGT for all platforms by just clicking a link. We also need to separate the public version docs with the development version docs on the website instead of showing the development version all the time, which is seriously confusing users of the public builds of the engine.

	anticheat:
		We have some pull requests open regarding anticheat particularly on windows, those should be finished up and merged.

	Angelscript version checks:
		One issue we run into relatively often is that someone will try building NVGT from source and then will report bytecode load errors to us because they're running from an outdated stub or something similar. We should encode the Angelscript version string into the bytecode payload and verify it upon application load. This way if bytecode gets attached to the wrong stub, we can catch it with a graceful error message instead of showing some random bytecode load aerror.

	get_last_error():
		One area of NVGT that still needs heavy improvement is error handling. Some things use exceptions, some libraries have a get_error function, some things may use the backwards compatibility function get_last_error() etc. We need to find a way to unify this as much as possible into one system.

	library object:
		NVGT does have a library object similar to BGT which allows one to call into most standard dlls. However NVGT's library object is still rougher than BGT's and could do with some work, particularly we may switch to libffi or dyncall or something like that. This object in nvgt is so sub-par because the engine's open source nature combined with the c++ plugins feature deprioritised the fixing of this system to the point where it remained broken beyond the official prerelease of NVGT. The library object functions, but one may have an issue for example when working with various types of pointers.

	dictionary serialization:
		Currently, NVGT's dictionaries can only serialize bool, int, double and string. This is highly limiting. In Angelscript we can traverse the properties of any class instance and even read the properties from it. Therefor, at least so long as a class inherits from a certain base of some kind, the user should be able to serialize and deserialize full objects including graceful handling for class structures changing over time.

	force_key methods:
		A rare request is that we add bgt's force_key_down/force_key_up methods and friends to the engine, this is a good idea and we will do so. We are very close now with the new simulate_key_down and simulate_key_up functions, the only difference between these and the force methods is that the player still controls the keyboard E. simulate arrow down then real arrow down followed by real arrow up means arrow released, where as with bgt force methods the arrow key would stay forced down until manually reset.

	nvgt.dll:
		A very long-term goal is to export as much of the engine as possible into a cdll so that it's features can be integrated into anyone's programs in other languages.

	Very basic graphics:
		No work has been done here yet, but a goal of ours is to utilize sdl / sdlttf to allow audio game developers to draw at least basic text to the screen, if not much more.

	wrap more of Poco:
		Poco is a massive library with many useful classes that touch on many areas, from networking to a process launcher to our datetime facility to several other utilities. We want to wrap much more of this library, including but not limited to:
		* configuration facility, it supports various formats and is extensible.
		* SMTP, we can send emails directly from nvgt game servers if we wrap this.
		* more sockets, we've registered raw tcp socket and web_socket, we should register the datagram socket, dialog socket, raw socket and more, though some are pretty low priority.
		* ProcessRunner and pipes or at least something similar to it, people want to capture process output.
		* Logging facility potentially. Do we need it or do I just think it's cool?
		* Notification queues for interthread communication.
		* UUID, I've held off because it forces us to use their bsd3 licensed random number generator but it could still be useful.

	audio_form rewrite:
		While the current audio form will always be here for backwards compatibility, it is unfortunately very old code which keeps gaining appendage after appendage after hack after hack in order to keep working propperly. It is a nightmare to integrate into a game translation system, control objects are private, it's just getting hardre to maintain. We need to provide a new audio form instead of encouraging people to use the existing one. Someone has started working on one, but it is in early stages given my last update on the situation.

	input_box on Linux and Mobile:
		The input_box function is currently only implemented on Windows and MacOS, we need to get it working on all platforms.

	reactphysics3d:
		We're working on integrating a physics engine into NVGT for much more advanced games. There are many of these out there, we chose reactphysics3d because it builds on all of our desired platforms, it's not utterly massive, it seems to contain many of the features we can imagine needing, and it does not come with a huge learning curve. Much work has been put into implementing reactphysics, though  most of it is on a different branch at this time called rp3d.

	tts_voice parameter scaling:
		In bgt, tts_voice parameters went from -10 to 10 for rate and pitch, and -100 to 0 for volume. However in NVGT, this currently differs per platform. We need to set up a couple of conversion functions that make bgt's old values work across the board, though possibly in floatingpoint so as to give NVGT users more precision if they desire it.

	package managers:
		This is a long-term goal, but consider figuring out how to create an apt repo / get on homebrew / other package managers for easy installs of NVGT.

	IOS:
		NVGT now has Android support, leaving one more target platform which is IOS!

	MacOS Code Signing:
		This is likely to be fixed very soon: Currently, it is not possible to code sign MacOS app bundles generated by NVGT, and the binaries in them do not have an executable bit set at this time. Once these issues are resolved, a vast portion of remaining MacOS issues will be resolved.

	Authenticode signing for official installers:
		Currently nvgt's official installers are unsigned and I am hoping to fix that.

	Android touchups:
		Android support is getting pretty stable, but there are a couple of areas that still need improvement. We still need to finish asset management, and so until that is done you must embed a pack file into your app for sounds to be played. More improvements to tts_voice will follow as well, you can only use the default voice at this time and parameter ranges may need adjustment. The gesture support though functional is still very young, and needs more testing and expansion.

	Code improvements:
		This section specifically lists tasks we wish to perform related to NVGT's c++ code, repository, or other backend. Other than performance, items here should rarely effect application functionality but are instead designed to allow developers to engage in a bit less painful wincing when reading or contributing to NVGT's code.

	types.h:
		Due to lack of experience towards the beginning of this project's development, our types are currently all over the place. There are random and completely unneeded mixes of asUINT, uint32_t, unsigned long, unsigned int, Poco::UInt32 and more all over the project. Instead we need to create types.h that define preferably types similar to angelscript (uint8, uint16, uint32, uint64) which we can use in the project, or at least we need to choose an existing set like Poco's or Angelscript's and stick with it consistently.

	Naming of globals:
		Along the same line, partly due to initial closed source intentions and also partly do to the use of sample Angelscript code, some of NVGT's global symbols are not named ideally. The best example right now is g_CommandLine vs. g_command_line_args. We need to decide on a scheme and stick to it unless forced by a dependency, and then do a quick symbol renaming session in vscode.

	Rewrite system_fingerprint.cpp:
		Currently we are using parts of an apache2 licensed library for system fingerprint generation. Not only is it a bit rough but it also uses several architecture specific assembly instructions at times when we probably don't need any. We should rewrite this to use our own system instead comprised of calls into Poco, SDL and other libraries that can return various bits of system information, or at the very least find a solid tiny dependency that can handle it for us. Legacy functions will still be provided.

	increased plugin security:
		Currently, one can achieve mischief by replacing a game's plugin dll with a hacked or modified version which uses the plugin's access to the underlying Angelscript engine to access sensative information within a game's loaded bytecode. While in the end we cannot prevent this entirely (it's a given downside when developing a plugin system like this that is still extensaable), but what we cann do is make it annoying. A pull request is already in progress which attempts to add cryptographic signatures to the plugin dlls loaded by a game which are verified by NVGT, at least applying an extra layer of annoyance/difficulty to anyone's attempt to go hacking around with an NVGT plugin, and it just needs some cleaning up/a bit of fixing before we can merge it. Anyone who is extremely worried about this will need to build NVGT from source with staticly linked plugins.

	Automated process to build platform specific dev packages:
		Currently when someone wants to build NVGT, they download something like windev.zip or macosdev.tar.gz or droidev.zip etc. That is helpful because many people who are in the middle between something like NVGT and getting started with C++ probably only want to change a few lines of NVGT and rebuild it. It's also helpful for speeding up our CI. However the problem is that these packages are manually maintained, when in reality we should be using something like vcpkg to set these up via a CI action of some sort.

	rewrite random.cpp somewhat:
		At the moment, I directly register the c interfaces of the public domain rnd.h library we are using with Angelscript. This however has created some undesirable issues, for example it's not possible to select a default rng, array.random() has several overloads for each type because they don't have a base class etc. Instead, create a base random interface with whatever generaters we want inheriting from it, then allow the user to select the default generator instance that the random() function uses.

	test cases:
		NVGT has a test runner, but it only tests a minimal subset of the engine at this time. Once we have more tests, we will hook the test runner up to the CI so we can instantly see if something goes wrong with the engine.


	Changelog:
		This document lists all major changes that have taken place in NVGT since we started keeping track.

	New in 0.89.1-beta (10/09/2024):
		* Fixes copying of shared libraries while bundling on MacOS which was resulting in an assertion violation from Poco.
		* Fixes MacOS not changing to the directory of scripts launched from finder or the open command.
		* Hopefully we've finally resolved all of the MacOS includes resolution issues, there should be no more manually adding include paths to your scripts or command line invocations!
		* Fixes the enter key not pressing the default button in audio form lists.
		* Linux cross compilation should work again, the Linux build script was pulling down the wrong version of Angelscript.
		* `#pragma platform` has been deprecated, you should use the command line / menu options / UI to specify platforms instead.
		* Adds preprocessor macros to check platform! You can surround code in `#if android ... #endif` for example, or `#if_not windows ... #endif` as well. You can even use this to specify custom includes or pragmas just on certain platforms.

	New in 0.89.0-alpha (10/09/2024):
		* NVGT has now switched to using InnoSetup for it's installer, adding several new options such as adding to path, start menu icons etc.
		* Added the script_function::get_namespace() method as well as script_function::retrieve() which takes a native function handle derived from a funcdef.
		* Including pack files via the `#include` directive has been removed in favor of being able to include scripts of any extension, you should use the `#pragma embed` directive to do this instead.
		* NVGT games now run on intel as well as arm mac computers!
		* The calendar object is now registered as a reference type with Angelscript meaning it now supports handles, for BGT backwards compatibility. The other datetime classes are still value types.
		* Added bool sdl_set_hint(const string&in hint, const string&in value, sdl_hint_priority priority = SDL_HINT_NORMAL) and string sdl_get_hint() functions, allowing the user to customize over 200 different SDL options from screen orientation to the video backend used and many more.
		* Adds atomic types, or in other words more concurrency primitives.
		* The array class can now work with negative indicies to access elements starting at the end of arrays, For example `array[-1]` now returns the last element in the array.
		* Registered SYSTEM_PERFORMANCE_COUNTER and SYSTEM_PERFORMANCE_FREQUENCY properties as well as nanosleep and nanoticks functions, still needs documentation.
		* ini.nvgt include is now free of bgt_compat! Added quick test for it.
		* Fixed a bug in instance.nvgt include regarding automatic mutex name generation.
		* The compiler now includes a status window that continues getting updated with messages reporting what is currently happening during compilation.
		* Adds refresh_window() function, allowing manual pulling for windows events encase you want to use your own sleeping methods.
		* Adds input_forms.nvgt include with functions to quickly retrieve information using an audio form, test/example in test/interact/test_input_forms.nvgt.
		* Introduce new menu system based on audio form, in menu.nvgt. As a result, the old dynamic menu has been renamed to bgt_dynamic_menu.nvgt to make it clear what one is legacy.
		* Registered some new vector functions such as cross, length2, distance and distance2 etc.
		* datastream.available is now an uint64 rather than int.
		* The ghost settings object in bgt_compat.nvgt was removed, as a new settings.nvgt include was contributed to the project! While the new settings object does not currently support the windows registry, this is made up for by several useful features such as encryption, data format selection, default values and more. You can read the top of settings.nvgt for a full list of changes. The new settings.nvgt script is `#included` by default in bgt_compat to avoid any compatibility issues.
		* Though this effort is still somewhat on going, most parameters of NVGT's builtin functions and methods have been properly named. Not only does this mean that passing named arguments works, but it is also a small step closer to a VSCode or similar plugin.
		* screen_reader_output/speak now have their interrupt booleans set to true by default.
		* Though a more advanced (multi-selection / non-blocking) API for this will be added in the future, simple open_file_dialog, save_file_dialog, and select_folder_dialog functions have now been added.
		* There are now 2 new functions which have yet to be documented, bool simulate_key_down(uint key) and bool simulate_key_up(uint key). These directly post SDL keydown/up events to the event queue.
		* Any key_up events with matching keycodes as any key_down event in the same frame will now be moved to the next frame. This previously only happened with a few specialized events (namely voice over arrow keys and windows clipboard history), but now this is done in all scenarios meaning that the Mac touch bar and other on screen or touch based keyboards should now function properly.
		* adds is_console_available function.
		* Fixed a serious bug in pack::add_memory that caused adding large files to usually fail.
		* Adds the `string generate_custom_token(int token_length, string characters);` function to token_gen.nvgt include.
		* Add virtual_dialogs.nvgt include.
		* Add datastream.sync_rw_cursors property (true by default).
		* tts_dump_config and tts_load_config in speech.nvgt include can now save screen reader usage setting.
		* ADDED ANDROID PLATFORM SUPPORT, NVGT RUNS ON MOBILE! This includes gesture detection (touch.nvgt include or query_touch_device function), screen reader speech through Android's accessibility event API, and android TextToSpeech engine support through  the tts_voice class! The support is still young and there are many improvements still to be made (only the default tts voice can be used right now and the only way to embed sounds is by using the `#pragma embed pack.dat` feature for example), but even running small nvgt scripts from source is possible at this point with NVGT's Android runner application, and one-click APK bundling is possible!
		* NVGT now has a bundling facility! It can create .apk packages for android (assuming the needed android tools are available), it can create MacOS app bundles on all platforms though only a .dmg on Mac (.app.zip on other platforms), and it can copy Windows/Linux libraries into place as well as bundle your asset files like sounds, readme and changelog all to create a fully distributable package in one click! It can even run custom prebuild and postbuild shell commands in case the bundling facility isn't quite doing enough for your needs. More information is in the compiling for distribution document. This means as an aside that NVGT's compiler application is significantly larger, as it must include the MacOS and Linux libraries on windows, the Windows and Linux libraries on Mac etc for the purposes of creating a fully functional app bundle no matter what platform it is compiled on.
		* The compile script submenu has changed, it now contains options to compile for all platforms you have stubs for!
		* Trying to embed a pack that doesn't exist no longer makes NVGTW.exe silently exit.
		* Added the is_window_hidden function.
		* NVGT's UI application now has a launcher dialog instead of informing the user that no input file was provided. You can run scripts, compile scripts including platform selection, or do a few other things such as viewing the command line arguments/version information.
		* The directory_delete function now includes a recursive boolean argument.
		* Upgrades NVGT to SDL3!
		* Switched to UniversalSpeech static, no more Tolk.dll required for screen reader output!
		* form.nvgt modifications:
		    * The `form::is_disallowed_char` method will now always return true if the character is not allowed. This also automatically checks whether the control index is being used as either blacklist or whitelist, the `use_only_disallowed_characters` parameter.
		    * Pasting text will now fail if the clipboard contains disallowed characters.
		    * Fixes a bug regarding the go to line dialog and the column being set 1 character to the left from where it should be.
		    * Adds the ability to create custom control type labels.
		    * Added Link control type.
		    * If you alt tab to the application from another program while in an audio form, it reannounces the current focus control.
		    * Added Floating points support in sliders and Fixed home/end bug with sliders.
		    * Adds set_list_properties method.
		    * Added word deletion.
		    * You can now serialize the `audioform_keyboard_echo` property.
		    * Once an input field is popped out, You can use the F2 key to change between echo modes. The changed echo mode will be spoken out loud, as well as updating the `audioform_keyboard_echo` property.
		    * The `reset` method now takes a new parameter, true by default, which resets the form's echo mode to default. This is useful to retrieve the value of the `audioform_keyboard_echo` property.
		    * The go to line functionality can now be used on non multiline input fields as well. In non multiline fields, the line number field will be invisible.
		    * Added a new method that allows you to toggle the go to line functionality. `bool set_enable_go_to_index(int control_index, bool enabled);`
		* Added a new parameter in sound_pool's `update_listener_3d` function called `refresh_y_is_elevation` (bool) which toggles whether the sound pool should refresh the global `sound_pool_default_y_is_elevation` property. This makes it possible to constantly change the global property for the sound elevation.
		* token_gen.nvgt can now generate different token types, see the `token_gen_flag` enum.
		* Add string::is_whitespace method.
		* Fix small memory leak in pathfinder due to not releasing callback data reference on path calculation failure.
		* Add SCRIPT_BUILD_TIME property.
		* Fixes missing const qualifier in mixer::set_fx which was causing random memory corruption on some systems and in some situations.
		* Slightly improves include path resolution when running nvgt scripts on Macos.
		* Speed improvements to find_files and find_directories especially on windows.
		* Datastreams can now be implicitly created from strings.
		* Add var::opCmp method.
		* Documentation:
		    * Updates compiling for distribution tutorial to talk about bundling facility.
		    * Talk about Angelscript addons and delayed dll loading in plugin creation tutorial.
		    * Documents TIMEZONE_* properties and refresh_window() function, update wait.nvgt to indicate that it calls refresh_window().
		    * Document settings.nvgt include.
		    * Document the number_speaker.nvgt include.
		    * Partially document timestamp class, still needs operators and maybe a couple other things.
		    * Completely documented thread_event class as well as concurrency enums.
		    * update datetime property examples to say "set thing" instead of "current thing," document julian_day property.
		    * Documents timestamp_from_UTC_time function.
		    * Documents thread_current_id and thread_yield.
		    * documents sound_global_pack property.
		    * Various minor polishing for a few functions in data manipulation, audio, and user interface.
		    * documented set_application_name, focus_window, is_window_hidden, and get_window_text. Also removed joystick_count from the documentation as it was an old SDL2 function.
		    * Update pack documentation to note a confusion with get_file_offset vs. offset parameter as well as fixing inconsistent parameter name in read_file.
		    * Added many more docs for the pack object, documented file_copy function.
		    * Updates toolkit configuration tutorial to include all new options.
		    * Added a note in the introduction topic that indicates the incomplete state of the docs.
		    * Update contributors.
		    * Array: `insert_at`, `remove_at`, most other methods with their examples.
		    * dictionary, almost everything accept the indexing operator.
		    * Environment global properties: `PLATFORM`, `PLATFORM_DISPLAY_NAME`, `PLATFORM_VERSION`, `system_node_name`, and `system_node_id`.
		    * file datastream, including write method in the datastream documentation.
		    * Token Gen documentation updated, including its enum constants.
		    * Adds a not yet complete memory management tutorial.
		    * Todo updated.
		    * Fixed return type of network::connect and resolve syntax error in tts_voice constructor reference, other various syntax and formatting.
		    * Audio game development tutorial has been slightly updated.
		    * audio_form documentation is now mostly complete.

	New in 0.88.0-beta (07/01/2024):
		* several improvements to the audio form:
		    * fix the go to line dialog, it had broken a few versions ago when converting the form to no longer need bgt_compat.
		    * adds method `bool is_disallowed_char(int control_index, string char, bool search_all=true);` which checks whether the given characters are allowed. It is also possible to make the search_all parameter to true or false to toggle whether the method should search to match full text, or every character. You can also omit the control_index parameter, in which case the method is used internally in the control class.
		    * adds method `bool set_disallowed_chars(int control_index, string chars, bool use_only_disallowed_chars=false, string char_disallowed_description="");` which sets the disallowed characters in a given control. Setting the use_only_disallowed_chars parameter to true will restrict to use only characters that have been set into this list. The char_disallowed_description parameter is also optional and sets the description or the text to speak when the user types the character that isn't allowed. Default is set to empty, meaning there will be silent when the user types the not allowed character. Setting the chars parameter to empty string will clear the list, thus setting to its original state.
		    * Adds methods is_list_item_checked and get_checked_list_items, also made the get_list_selections function return a handle to the array.
		    * Add a type of list called tab panel. This does not include the ability to assign controls to each tab automatically right now, but instead just the facility to create a list that acts more like a tab control.
		* adds blocking functions to and exposes the pause functionality in the music manager include.
		* adds a boolean to sound.play (true by default) which controls whether to reset the loop state on sound resume, so it's now possible to pause a looping sound and resume it without knowing whether the sound loops.
		* add functions to var type such as is_integer, is_boolean, is_string and more to determine what is stored in the var, also var.clear().
		* Added `sound_pool_default_y_elevation` property into sound_pool. This property will be useful to set default `y_is_elevation` property of each sound pool without having to change later for every sound pool you declare. Note, however, that it will only work for sound_pools declared after the variable is set, for example a global sound_pool variable will likely initialize before any function in your code could set the property.
		* improve passing arguments to async calls.
		* Fix critical bug if using network_event by handle that could allow the static none event to get value assigned to! Also better type info caching during the creation of some arrays.
		* improve the speed of stream::read() when the stream size can be determined, which can be done automatically for files.
		* users should no longer need to install libgit2 and openssl in order to run NVGT games on MacOS!
		* Implement the recently contributed AVTTSVoice class into nvgt's tts_voice object, meaning we now have AVSpeechSynthesizer support for MacOS in NVGT! Temporary caveats:
		    * Though this works well enough for people to start playing with, it still needs more testing. In particular, too much switching of voices followed by speech could cause a tts_voice object to break, calling refresh would probably fix it. Exact condition to reproduce unknown.
		    * We also still need to figure out how to clamp the MacOS rate/pitch/volume parameters into the -10 to +10 that is typically used in the tts_voice class, so parameter adjustment may be rough for a short time.
		    * The contributed class provides methods to handle languages, but the nvgt tts_voice class does not yet have these, as such voices just have raw names for now until we brainstorm more API.
		    * For now this class speaks using the operating system directly and does not interact with bass, and so speak_to_memory is not yet supported. This will be addressed as the AVSpeech API does give us methods to do this.
		* It is now possible to open .nvgt scripts directly within finder with the MacOS app bundle!
		* array.insert_last can now take a handle to another array.
		* Documentation:
		    * major spell checking session as well as some formatting.
		    * array::insert_last method.
		    * sound::loaded_filename method.
		    * string::format method.
		    * get_characters function
		    * idle_ticks function.
		    * mutex class
		    * concurrency article
		    * Game programming tutorial updates
		    * Clarifies some information in the distrobution topic.
		    * Include direct links to MacOS and Linux build scripts.
		    * Minor updates to contributors topic.
		    * Updates to the todo and bgt upgrading topics.

	New in 0.87.2-beta (06/17/2024):
		* Hopefully removed the need for the user to run `xattr -c` on the mac app bundle!
		* Fix an accidental UTF8 conversion issue introduced into screen reader speech that took place when implementing speech dispatcher.
		* NVGT will no longer fail if a plugin pragma with the same plugin name gets encountered multiple times.
		* Fix minor error in bgt_compat's set_sound_storage function where the filename stored was not caching for display properly.
		* Improve the downloads page to include release dates and headings for old versions as well as file sizes, yes we do still intend to integrate with github releases.
		* Fix the broken SCREEN_READER_AVAILABLE property on windows.
		* New documentation on compiling a game for distribution.

	New in 0.87.1-beta (06/16/2024):
		* This patches an issue with the speech dispatcher support on linux where calling screen_reader_unload() would cause a segmentation fault if it had not loaded to begin with due to no libspeechd being available.
		* Fixed a severe lack of debugging information issue where if a plugin could not load on linux, no error information was printed and instead the app silently exited.
		* Fix an issue where some error messages that printed to stdout were not following up with a new line.
		* Another enhancement to the network object which avoids an extra memory allocation every time a network_event of type event_none was created.

	New in 0.87.0-beta (06/16/2024):
		* The jaws keyhook works even better than it did before and is nearing full functionality, though you may still need to alt+tab a few times in and out of the game window if JAWS restarts.
		* Various improvements to the network object:
		    * added packet_compression boolean which allows the user to toggle enet's builtin packet compressor allowing nvgt network objects to be backwards compatible with BGT if this is enabled!
		    * Added packets_received and packets_sent properties.
		    * network.destroy takes an optional boolean argument (flush = true) which can make sure any remaining packets are dispatched before the host is destroyed.
		    * bytes_received and bytes_sent properties no longer wrap around after 4gb due to enet tracking that information using 32 bit integers.
		* Add support for speech dispatcher on Linux and other BSD systems!
		* You can now use the directive `#pragma embed packname.dat` as an alternative to `#including` your pack files.
		* Fix broken quoted strings when handling pragma directives, resolving issues with the `#pragma include` option.
		* Mostly finishes making it possible to configure various Angelscript engine properties as well as other NVGT options using configuration files, just needs polishing.
		* bgt_compat's string_to_number function now trims whitespace to comply with bgt's standard, our new faster parse_float doesn't do that anymore in the name of performance.
		* Fix issues with sound::push_memory():
		    * actual audio files can be pushed to the function without modifying any other arguments other than the data string.
		    * Calling sound.load() with a blank filename is no longer required before push_memory functions.
		* Some polishing to the Angelscript integration:
		    * If a global variable fails to initialize, the exception that caused it is now properly shown in the compilation error dialog.
		    * There should hopefully be no more cases where a compilation error dialog can show up after a script executes successfully and throws an exception. Instead if the user enables the warnings display, they should properly see warnings before the script executes.
		    * Set up the infrastructure to be able to store a bit of extra encrypted information along side the bytecode payload in compiled programs, for now using that to more securely store the list of enabled plugins as well as the serialized Angelscript properties.
		    * Scripts no longer compile if they do not contain a valid entry point.
		* Updated to the latest Angelscript WIP code which resolves a bytecode load error that had been reported.
		* Revert the code changes to mixer::set_fx back to NVGT's first public release as the refactor did not go well and continued introducing unwanted side effects.
		* Fixed bugs in find_directories and find_files on Unix platforms, the functions should now behave like on windows.
		* Adds idle_ticks() function (works on windows and MacOS at present) which returns the number of milliseconds since the user has been idle.
		* Update Angelscript's script builder addon which makes it possible to use unicode characters in script include paths.
		* Add multiplication operators to strings, for example `string result = "hello" * 10;`
		* There is a new way to list files and directories, a function called glob. Not only can it return all files and directories in one call, but you can even provide wildcards that enter sub directories. The function is documented in the reference.
		* New additional version of json_array and json_object.stringify() which takes a datastream argument to write to.
		* json_array and json_object now have get_array and get_object methods that directly return casted json_array@ or json_object@ handles.
		* Added default_interrupt property in the high level Speech.nvgt include to set default interrupting for all the speech.
		* Documentation: configuration tutorial, much of tts_voice object and other minor improvements.

	New in 0.86.0-beta (06/08/2024):
		* running nvgtw.exe or the mac app should now show a message box at least rather than silently exiting.
		* Improves the functionality of the JAWS keyhook. We likely still have more to go before it's perfect, but it is at least far better than it was before.
		* pack::open is now set to read mode by default and will try closing any opened pack rather than returning false in that case.
		* Added sound.loaded_filename property to determine the currently loaded filename of a sound object.
		* Added string.reserve() function.
		* Added  get_window_os_handle() function.
		* Fix issues in sound::set_fx in regards to effect deletion.
		* NVGT's datetime facilities now wrap Poco's implementations. Documentation is not complete, but the 4 new classes are datetime, timestamp, timespan, and calendar (which wraps LocalDateTime) in Poco and is called calendar for bgt backwards compatibility. Global functions include parse_datetime, datetime_is_leap_year and more, and all classes include a format method to convert the objects into strings given a format specifier.
		* Converted most filesystem functions to wrap Poco's implementations.
		* Fix potential issue with network where packets don't destroy on send failure.
		* Added string_create_from_pointer to library functions.
		* Though the sound pool will soon be superseded by better methods of handling sounds, it has received various improvements nevertheless:
		    * Add y_is_elevation property. When this is set to false, the positioning works as normal, E.G. x is left/right, y is back/forward, and Z is up/down. When it's set to true, y is now up/down and Z is back/forward. This is useful for 2D platforming games for example, or games where y is up/down and Z is back/forward rather than the reverse. At some point this will be built into the engine so that it can be used on sound objects directly.
		    * Fix a bug when using the new sound_default_pack global property.
		    * Cleaned up the constructors and play_extended functions a bit.
		    * bgt_compat.nvgt is no longer required to used this include.
		* Adds a new function to dynamic_menu (add_multiple_items) that accepts an array of arrays. Each subarray must have at least 1 element. The first element is the text, the second element is the name. (#31)
		* Fix a couple of issues in the url_post implementation.
		* Register the Angelscript math complex addon.
		* fix missing include of bgt_compat in dynamic_menu.nvgt
		* fix number_to_words implementation appending null character to the end of it's output string
		* Adds a set_sound_storage function to bgt_compat.nvgt which takes advantage of the new sound_default_pack property.
		* Still a long long ways to go, but minor docs updates and a couple of new test cases.

	New in 0.85.1-beta (06/03/2024):
		* The restart method on timers will now unpause them.
		* Dramatically increase the speed of floating point string parsing.
		* The form and speech includes no longer require  bgt_compat!
		* adds sound_default_pack property.
		    * For example, you can now create a pack object, execute the expression @sound_default_pack = my_pack; and from that point all sounds in the engine will use the default pack you have set unless you explicitly override it.
		* Modified number conversion in sound mixer classes to use more efficient string -> number handling
		* Fixed typo in doc for BSL license
		* Start on the documentation for the pack object
		* include form.nvgt in the speech.nvgt include by default.
		* Add includes/number_speaker.nvgt (a port of bgt's class); makes it very easy to self-voice numbers!
		* There are now a few more test cases as well as the beginnings of a benchmarking framework which will begin leading to several speed improvements such as the floating point processor mentioned above!

	New as of 05/31/2024:
		* complete rewrite of NVGT entry point to use a Poco application. This includes much cleanup and organization and adds several features:
		    * There is now proper command line parsing, help printing, config file loading for when we want to use it, and more.
		    * This also introduces the change that on windows, there is now nvgt.exe and nvgtw.exe that gets built, one with the windows subsystem and one without.
		    * Script files and nvgt's binary will check for config files with the basename of file + .ini, .properties, or .json.
		* The above changes mean that we now implement the Angelscript debugger since the nvgt compiler is now always available in console mode.
		* NVGT now uses symantic versioning, for example 0.85.0.
		* Fixed script return code.
		* NVGT finally has a cross-platform installer (NSIS on Windows and a .dmg file on macOS).
		* The timer class once present in `bgt_compat.nvgt` is finally in the core of the engine!
		* it is now possible to embed packs into executables! 
		* The way Windows binaries load has been changed, meaning that UPX or any other binary compressor that supports overlays can now be used on your compiled NVGT binaries!
		* The timer resolution should be much more accurate on Windows.
		* Added a new, optional `uint timeout` parameter to the `network.request()` method.
		* Improved documentation.

	New as of 05/25/2024:
		* Wrapped `thread_pool` and `thread_event` from Poco.
		* New function: `script_engine_dump_configuration()`.
		* We now have an ftp_client class.
		* raw memory allocation functions and memory_reader/writer datastreams to use them.
		* May need more testing, but added the `async` class, for easily running a function on a thread and getting its return value.
		* Many more docs.

	New as of 05/08/2024:
		* Added a plugin to send notifications to systemd on Linux.
		* Many more docs.
		* `string.split()` should now reserve memory properly.
		* Wrapped Poco for HTTP/HTTPS requests.
		* Fix ancient bugs in soundsystem including too many functions registered as const and `close()` returning true on no-op.

	New as of 04/18/2024:
		* The var type now has PostEnc and PostDec operators.
		* UTF8 fixes: sound.load, and compiled applications can now execute if they contain non-english characters in their filenames.
		* All code that I wish to share has been forked into what will hopefully be nvgt's long-standing repository which will eventually have it's privacy status switched to public!
		* NVGT now has a build system! I know it's not the fastest one around, but needing a  middle ground between learning even more new things and using what I already know, I chose SCons purely because of the familiar pythonic environment and not needing to learn yet another new set of syntax rules. I'm just glad we're no longer building the engine using a series of shell scripts!
		* Added basic steam audio reverb integration! It needs a lot of work and is far from being production ready (seriously this could slow your game to a crawl until I'm done with this), but nevertheless it is still around for testing!

	New leading up to 02/20/2024:
		* NVGT now finally has a documentation structure!
		* Unifying the file object into stream classes.

	New as of 01/06/2024:
		* Misc sound system improvements (including the ability to set the pitch of a sound as low as 5).
		* Fix memory leak.
		* Remove sound_environment class for now.
		* Should load bassflac.dll and bassopus.dll if present
		* JSON Support.
		* Better multithreading support with more primitives.
		* More functions in the string class.
		* New methods of operating system detection.
		* Instance class removed from engine and replaced with include/instance.nvgt which wraps a named_mutex.
		* Regular expressions now use PCRE2 via poco including lower level regexp class.
		* Alert and question now use SDL message boxes (not sure what I think of this).
		* Other misc changes.

	Note for changes before 01/06/2024:
		Changes were roughly untracked before this time, but there is a rather large list of somewhat sorted changes below as committed to nvgt_bin (a repository where the NVGT testers could access and test NVGT). These are sorted by month where possible to make them easier to sort, but keep in mind that commits to nvgt_bin usually occurred all at once so that building NVGT was easier for all platforms. As such, expect these lists (while somewhat sorted) to become rather large! Additionally, some of these changes may be ambiguous due to being based off of nvgt_bin's commit messages only. At this time, it was assumed anyone using this engine had direct contact with Sam to ask questions.

	New as of 12/10/2023:
		* Using more poco libraries including basic json implementation.
		* Improvements to the sound crash.
		* Fixes instances where the engine was not handling UTF8 properly.
		* Performance increases such as updated hashmap and making network object faster.
		* First and probably unstable implementation of a plugin system.
		* Attempts to improve 3d pathfinding.
		* Example of subscripting.
		* More misc changes.

	New as of 07/24/2023:
		* Switch to sdl2 for keyboard/windowing (new key_repeating / get_preferred_locales() / urlopen() functions as a result).
		* Switch random number generators (see random_* classes).
		* Fixed random_get/set_state().
		* Now using Poco c++ libraries for various encode/decode/hash operations as well as better TIMEZONE_* properties and more (thus dropped cppcodec).
		* UTF8 support.
		* Multithreading support including mutex class.
		* Library object.
		* Basic msgpack support (see packet and string.unpacket functions).
		* md5/sha1/sha224/sha384 hashing added as well as HOTP function (see TOTP.nvgt example).
		* libgit2 support.
		* Bytecode compression (#pragma bytecode_compression 0-9).
		* Multiple windows stubs including Enigma Virtual Box.
		* Reduced sound crashes
		* Resolved a tts_voice crash.
		* More misc.

	New as of 04/24/2023:
		* Improvements to db_props include.
		* New system information functions for custom system fingerprint or error tracking.
		* Improvements to coordinate_map.
		* Subscripting can now compile to bytecode.
		* Fixed vector division scaling operators.
		* Improved reliability of timer queue.
		* Many more minor bugfixes.

	New as of 02/22/2023:
		* Fixes major rotation issue
		* Sqlite has more functions.
		* db_props updated.
		* Minor readme updates.
		* scriptstuff reference issue fixes.
		* Pathfinder micro speed improvement.
		* file_hard_link and file_hard_link_count.
		* More.

	New as of 01/17/2023:
		* Sans speech and windowing, NVGT runs on Linux!
		* Updated Bass/SteamAudio.
		* SQLite3 wrapper.
		* Improvements to subscript execution.

	New as of 10/21/2022:
		* Script_module and script_function classes.
		* Reduced sound crash.
		* speed improvements and more.

	New as of 09/20/2022:
		* Updated sound docs.
		* Added sound_default_mixer property.

	New as of 09/09/2022:
		* If you call the wait() function with long duration, the window no longer hangs.
		* Fix string_hash() issue.
		* Updated some BGT to NVGT gotchas.

	New as of 09/08/2022:
		* Fixes string_split().

	New as of 09/07/2022:
		* Upgrades SteamAudio to 4.11.
		* Reduces sound crash.

	New as of 09/02/2022:
		* Fixed bug in crypto.
		* Sound crash now much more rare.
		* String coordinate_map_area::type replaced with any@ coordinate_map_area::primary_data.
		* Other misc.

	New as of 08/20/2022:
		* print() function should now be lowercase again.
		* Many very minor fixes and improvements.

	New as of 07/01/2022:
		* Partially recoded pack streaming system to hopefully reduce sound crashes.
		* Various random speed improvements and fixes.

	New as of 06/30/2022:
		* Fixes a few speed issues with includes.
		* adds ini.list_wildcard_sections().

	New as of 06/02/2022:
		* Mostly works on making sound/pack more threadsafe.
		* Make ini loading robust.
		* Documents thread lock functions.

	New as of 05/26/2022:
		* Documentation and test for include/music.nvgt.
		* Updated readme a bit.
		* Working on sound callbacks.
		* Enabled bass_asyncfile for faster sound playback.

	New as of 05/22/2022:
		* Updated INI.

	New as of 05/21/2022:
		* sound.set_length() for streaming sounds.

	New as of 05/15/2022:
		* Fixed run function regarding filenames with spaces.
		* FTP uploads.
		* MLSD directory listings with internet_request.

	New as of 05/08/2022:
		* Bullet3 vectors.
		* size_to_string updates.
		* Other misc.

	New as of 04/26/2022:
		* Sound preloading.
		* byteshift encryption.
		* timer_queue.exists and timer_queue.is_repeating.
		* Minor speed improvements.



	Contributors:
		This file contains the names of anyone who has helped contribute to the NVGT engine in any significant way along with short descriptions of how these people have contributed. For a list of third party code used in the engine, you can view the Third Party Licenses topic. A huge thanks goes out to everyone listed here!

		* [Patrick W](https://github.com/braillescreen): some docs, build scripts, beta testing, and miscellaneous organization.
		* [Quin Gillespie](https://github.com/trypsynth): responsible for a growing number of API references documentation topics, beta testing.
		* [Rory Michie](https://github.com/RoryMichie): Has done much work on the NVGT user manual.
		* [Tyler Spivey](https://github.com/tspivey): rewrote the executable loader on Windows to work with packers such as UPX, helped make documentation look more presentable.
		* [Harry Min Khant](https://github.com/harrymkt): Wrote several documentation articles particularly in the references section, improvements to nvgt includes.
		* [Ethin Probst](https://github.com/ethindp): major optimizations to parts of the code, rewrote sound effect parsing, libspeechd and Android TTS engine integration, atomics, innosetup installer, docs.
		* [Ivan Soto](https://github.com/ivansoto0): Several major includes such as input_forms.nvgt, settings.nvgt, touch.nvgt etc as well as various improvements to smaller ones such as bgt_compat, input_forms and others.
		* [Valiant8086](https://github.com/valiant8086): Documentation on game distribution.
		* [Gruia Chiscop](https://github.com/GruiaChiscop): AVSpeech implementation.
		* [Hamza Ahmad](https://github.com/literary-programmer): Contributed virtual_dialogs.nvgt include, Improvements to form.nvgt.
		* [Abdullah Tepeli](https://github.com/colonel-official): Delete-by-word support for the audio_form include.
		* [Caturria](https://github.com/caturria): Much help with miniaudio implementation and bgt backwards compatibility, from a rewritten pack file with encryption to miniaudio vfs wiring to pathfinder/tts_voice fixes and more.
		* Beta testers including but not limited to Patrick Wilson, Quin G, Steven D, Lucas Brown, Liam Erven, DJWolfy, Lukáš Hosnedl, Jonathan859, Remi, Pragma and Day Garwood, without the valuable feedback and suggestions provided by these people NVGT would have never gotten this far.
		* Last but not least, nothing is worth maintaining or developing without users, and so thank you to everyone who uses this engine and gives it their feedback, time and attention!


	License agreement:
		NVGT - NonVisual Gaming Toolkit

		Copyright (c) 2022-2025 Sam Tupy

		[nvgt.dev](https://nvgt.dev)

		This software is provided "as-is", without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.

		Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

		1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
		2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
		3. This notice may not be removed or altered from any source distribution.


	Third party code attributions:
		This application may use any amount of the following copywrited code or components, though it may not use all of them or may use some of them minimally.

		This document may not serve as a complete reference of all copywrited material used in this application, and thus it's distribution should be checked for other similar documents to collect a complete list of copywrited content used in this software.

	Summary of components:
		The following is a convenient listing of third party material that this application may use, the full text for each license can be found below the lists.

	ZLib licensed code:
		* [Angelscript - AngelCode Scripting Library](#zlib_angelscript)
		* [Blastspeak - SAPI5 text to speech library for windows](#zlib_blastspeak)
		* [bullet3 - physics library](#zlib_bullet3)
		* [Micropather - n-dimensional AStar pathfinding library](#zlib_micropather)
		* [SDL - Simple DirectMedia Layer](#zlib_sdl)
		* [TINYEXPR - Tiny recursive descent parser and evaluation engine in C](#zlib_tinyexpr)
		* [zlib - general purpose compression library](#zlib_zlib)

	Code in the public domain:
		* [blastlibs - number to words routine written by Philip Bennefall](#PD_blastlibs)
		* [pdjson - Public Domain JSON Parser for C](#PD_pdjson)
		* [rnd.h - random number generators written by Mattias Gustavsson](#PD_rnd)
		* [speech.h - single header RSynth derivative maintained by Mattias Gustavsson](#PD_rsynth)
		* [SQLite - database engine](#PD_sqlite)
		* [tiny-AES-c](#PD_TinyAES)

	Code released under the boost software license version 1.0:
		* [fast_float - fast floating-point parsing library](#BSL_fast_float)
		* [POCO - Portable Components for c++](#BSL_poco)

	Code released under an Apache 2.0 or similar license:
		* [Steam Audio](#apache_phonon)

	GNU (lesser/library/linking exception) general public licensed code (lgpl):
		* [libplist - apple property list handling](#LGPL_libplist)
		* [NVDA controler client](#LGPL_NVDA)

	MIT licensed code:
		* [cmp - msgpack serialization library](#MIT_cmp)
		* [enet library](#MIT_enet)
		* [Hammerspoon's input box dialog extension for MacOS](#MIT_hammerspoon)
		* [Ratas - hierarchical timer wheel](#MIT_ratas)
		* [UniversalSpeech - screen reader library for windows](#MIT_universal_speech)
		* [unordered_dense - fast unordered map and sets for c++](#MIT_unordered_dense)
		* [Zodiac's angelscript print function](#MIT_zodiac)

	BSD 2 or 3 clause licensed code:
		* [Code from the FreeBSD Project](#BSD_bsd)
		* [double-conversion](#BSD_double_conversion)
		* [libopus - opus audio codec](#BSD_opus)
		* [PCRE - Perl Compatible Regular Expressions](#BSD_PCRE)
		* [libvorbis - vorbis audio codec](#BSD_vorbis)
		* [wepoll - epoll support for Windows](#BSD_wepoll)

	ZLib licensed code:

	<a id="zlib_angelscript">Angelscript - AngelCode Scripting Library</a>:
		Copyright (c) 2003-2023 Andreas Jonsson

		This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.

		Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

		1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.

		2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.

		3. This notice may not be removed or altered from any source distribution.

		The original version of this library can be located at:
		http://www.angelcode.com/angelscript/

		Andreas Jonsson
		andreas@angelcode.com

	<a id="zlib_blastspeak">Blastspeak - SAPI5 text to speech library for windows</a>:
		Copyright (c) 2019-2020 Philip Bennefall

		This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.

		Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

		1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.

		2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.

		3. This notice may not be removed or altered from any source distribution.

	<a id="zlib_bullet3">bullet3 - physics library</a>:

		Copyright (c) 2003-2013 Gino van den Bergen / Erwin Coumans  http://bulletphysics.org

		This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.

		Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

		1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.

		2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.

		3. This notice may not be removed or altered from any source distribution.

	<a id="zlib_micropather">Micropather - n-dimensional AStar pathfinding library</a>:
		Copyright (c) 2000-2013 Lee Thomason (www.grinninglizard.com)

		This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.

		Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

		1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgement in the product documentation would be appreciated but is not required.

		2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.

		3. This notice may not be removed or altered from any source distribution.

	<a id="zlib_sdl">SDL - Simple DirectMedia Layer</a>:
		Copyright (C) 1997-2024 Sam Lantinga <slouken@libsdl.org>

		This software is provided 'as-is', without any express or implied warranty.  In no event will the authors be held liable for any damages arising from the use of this software.

		Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

		1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.

		2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.

		3. This notice may not be removed or altered from any source distribution.

	<a id="zlib_tinyexpr">TINYEXPR - Tiny recursive descent parser and evaluation engine in C</a>:
		Copyright (c) 2015-2020 Lewis Van Winkle

		http://CodePlea.com

		This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.

		Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

		1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgement in the product documentation would be appreciated but is not required.

		2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.

		3. This notice may not be removed or altered from any source distribution.

	<a id="zlib_zlib">zlib - general purpose compression library</a>:

		Copyright (C) 1995-2017 Jean-loup Gailly and Mark Adler

		This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.

		Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

		1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgement in the product documentation would be appreciated but is not required.

		2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.

		3. This notice may not be removed or altered from any source distribution.

		Jean-loup Gailly        Mark Adler
		jloup@gzip.org          madler@alumni.caltech.edu

	Code in the public domain:

	<a id="PD_blastlibs">blastlibs - number to words routine written by Philip Bennefall</a>:

		This is free and unencumbered software released into the public domain.

		Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

		In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	<a id="PD_pdjson">pdjson - Public Domain JSON Parser for C</a>:

		This is free and unencumbered software released into the public domain.

		Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

		In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	<a id="PD_rnd">rnd.h - random number generators written by Mattias Gustavsson</a>:

		This is free and unencumbered software released into the public domain.

		Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

		In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	<a id="PD_rsynth">speech.h - single header RSynth derivative maintained by Mattias Gustavsson</a>:

		This is free and unencumbered software released into the public domain.

		Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

		In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	<a id="PD_sqlite">SQLite - database engine</a>:

		The author disclaims copyright to this source code.  In place of a legal notice, here is a blessing:

		*   May you do good and not evil.
		*   May you find forgiveness for yourself and forgive others.
		*   May you share freely, never taking more than you give.

	<a id="PD_TinyAES">tiny-AES-c</a>:

		This is free and unencumbered software released into the public domain.

		Anyone is free to copy, modify, publish, use, compile, sell, or
		distribute this software, either in source code form or as a compiled
		binary, for any purpose, commercial or non-commercial, and by any
		means.

		In jurisdictions that recognize copyright laws, the author or authors
		of this software dedicate any and all copyright interest in the
		software to the public domain. We make this dedication for the benefit
		of the public at large and to the detriment of our heirs and
		successors. We intend this dedication to be an overt act of
		relinquishment in perpetuity of all present and future rights to this
		software under copyright law.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
		EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
		MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
		IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
		OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
		ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
		OTHER DEALINGS IN THE SOFTWARE.

		For more information, please refer to <http://unlicense.org/>

	Code released under the boost software license version 1.0:

	<a id="BSL_fast_float">fast_float - fast floating-point parsing library</a>:
		Boost Software License - Version 1.0 - August 17th, 2003

		Permission is hereby granted, free of charge, to any person or organization
		obtaining a copy of the software and accompanying documentation covered by
		this license (the "Software") to use, reproduce, display, distribute,
		execute, and transmit the Software, and to prepare derivative works of the
		Software, and to permit third-parties to whom the Software is furnished to
		do so, all subject to the following:

		The copyright notices in the Software and this entire statement, including
		the above license grant, this restriction and the following disclaimer,
		must be included in all copies of the Software, in whole or in part, and
		all derivative works of the Software, unless such copies or derivative
		works are solely in the form of machine-executable object code generated by
		a source language processor.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
		IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
		FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
		SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
		FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
		ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
		DEALINGS IN THE SOFTWARE.
	<a id="BSL_poco">POCO - Portable Components for c++</a>:
		Copyright (c) 2023, Applied Informatics Software Engineering GmbH.

		and Contributors.

		Permission is hereby granted, free of charge, to any person or organization obtaining a copy of the software and accompanying documentation covered by this license (the "Software") to use, reproduce, display, distribute, execute, and transmit the Software, and to prepare derivative works of the Software, and to permit third-parties to whom the Software is furnished to do so, all subject to the following:

		The copyright notices in the Software and this entire statement, including the above license grant, this restriction and the following disclaimer, must be included in all copies of the Software, in whole or in part, and all derivative works of the Software, unless such copies or derivative works are solely in the form of machine-executable object code generated by a source language processor.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	Code released under an Apache 2.0 or similar license:

	<a id="apache_phonon">Steam Audio</a>:
		Copyright 2017-2023 Valve Corporation.

		Licensed under the Apache License, Version 2.0 (the "License");
		you may not use this file except in compliance with the License.

		You may obtain a copy of the License at
		http://www.apache.org/licenses/LICENSE-2.0

		Unless required by applicable law or agreed to in writing, software
		distributed under the License is distributed on an "AS IS" BASIS,
		WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
		See the License for the specific language governing permissions and
		limitations under the License.

	GNU (lesser/library/linking exception) general public licensed code (lgpl):

	<a id="LGPL_libplist">libplist - apple property list handling</a>:
		Copyright (c) 2012-2023 Nikias Bassen, All Rights Reserved.

		Copyright (c) 2008-2009 Jonathan Beck, All Rights Reserved.

		This library is free software; you can redistribute it and/or
		modify it under the terms of the GNU Lesser General Public
		License as published by the Free Software Foundation; either
		version 2.1 of the License, or (at your option) any later version.

		This library is distributed in the hope that it will be useful,
		but WITHOUT ANY WARRANTY; without even the implied warranty of
		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
		Lesser General Public License for more details.

		You should have received a copy of the GNU Lesser General Public
		License along with this library; if not, write to the Free Software
		Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

		GNU Lesser General Public License
		=================================

		_Version 2.1, February 1999_  
		_Copyright © 1991, 1999 Free Software Foundation, Inc._  
		_51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA_

		Everyone is permitted to copy and distribute verbatim copies
		of this license document, but changing it is not allowed.

		_This is the first released version of the Lesser GPL.  It also counts
		as the successor of the GNU Library Public License, version 2, hence
		the version number 2.1._

	Preamble:

		The licenses for most software are designed to take away your
		freedom to share and change it.  By contrast, the GNU General Public
		Licenses are intended to guarantee your freedom to share and change
		free software--to make sure the software is free for all its users.

		This license, the Lesser General Public License, applies to some
		specially designated software packages--typically libraries--of the
		Free Software Foundation and other authors who decide to use it.  You
		can use it too, but we suggest you first think carefully about whether
		this license or the ordinary General Public License is the better
		strategy to use in any particular case, based on the explanations below.

		When we speak of free software, we are referring to freedom of use,
		not price.  Our General Public Licenses are designed to make sure that
		you have the freedom to distribute copies of free software (and charge
		for this service if you wish); that you receive source code or can get
		it if you want it; that you can change the software and use pieces of
		it in new free programs; and that you are informed that you can do
		these things.

		To protect your rights, we need to make restrictions that forbid
		distributors to deny you these rights or to ask you to surrender these
		rights.  These restrictions translate to certain responsibilities for
		you if you distribute copies of the library or if you modify it.

		For example, if you distribute copies of the library, whether gratis
		or for a fee, you must give the recipients all the rights that we gave
		you.  You must make sure that they, too, receive or can get the source
		code.  If you link other code with the library, you must provide
		complete object files to the recipients, so that they can relink them
		with the library after making changes to the library and recompiling
		it.  And you must show them these terms so they know their rights.

		We protect your rights with a two-step method: **(1)** we copyright the
		library, and **(2)** we offer you this license, which gives you legal
		permission to copy, distribute and/or modify the library.

		To protect each distributor, we want to make it very clear that
		there is no warranty for the free library.  Also, if the library is
		modified by someone else and passed on, the recipients should know
		that what they have is not the original version, so that the original
		author's reputation will not be affected by problems that might be
		introduced by others.

		Finally, software patents pose a constant threat to the existence of
		any free program.  We wish to make sure that a company cannot
		effectively restrict the users of a free program by obtaining a
		restrictive license from a patent holder.  Therefore, we insist that
		any patent license obtained for a version of the library must be
		consistent with the full freedom of use specified in this license.

		Most GNU software, including some libraries, is covered by the
		ordinary GNU General Public License.  This license, the GNU Lesser
		General Public License, applies to certain designated libraries, and
		is quite different from the ordinary General Public License.  We use
		this license for certain libraries in order to permit linking those
		libraries into non-free programs.

		When a program is linked with a library, whether statically or using
		a shared library, the combination of the two is legally speaking a
		combined work, a derivative of the original library.  The ordinary
		General Public License therefore permits such linking only if the
		entire combination fits its criteria of freedom.  The Lesser General
		Public License permits more lax criteria for linking other code with
		the library.

		We call this license the “Lesser” General Public License because it
		does Less to protect the user's freedom than the ordinary General
		Public License.  It also provides other free software developers Less
		of an advantage over competing non-free programs.  These disadvantages
		are the reason we use the ordinary General Public License for many
		libraries.  However, the Lesser license provides advantages in certain
		special circumstances.

		For example, on rare occasions, there may be a special need to
		encourage the widest possible use of a certain library, so that it becomes
		a de-facto standard.  To achieve this, non-free programs must be
		allowed to use the library.  A more frequent case is that a free
		library does the same job as widely used non-free libraries.  In this
		case, there is little to gain by limiting the free library to free
		software only, so we use the Lesser General Public License.

		In other cases, permission to use a particular library in non-free
		programs enables a greater number of people to use a large body of
		free software.  For example, permission to use the GNU C Library in
		non-free programs enables many more people to use the whole GNU
		operating system, as well as its variant, the GNU/Linux operating
		system.

		Although the Lesser General Public License is Less protective of the
		users' freedom, it does ensure that the user of a program that is
		linked with the Library has the freedom and the wherewithal to run
		that program using a modified version of the Library.

		The precise terms and conditions for copying, distribution and
		modification follow.  Pay close attention to the difference between a
		“work based on the library” and a “work that uses the library”.  The
		former contains code derived from the library, whereas the latter must
		be combined with the library in order to run.

	TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION:

		**0.** This License Agreement applies to any software library or other
		program which contains a notice placed by the copyright holder or
		other authorized party saying it may be distributed under the terms of
		this Lesser General Public License (also called “this License”).
		Each licensee is addressed as “you”.

		A “library” means a collection of software functions and/or data
		prepared so as to be conveniently linked with application programs
		(which use some of those functions and data) to form executables.

		The “Library”, below, refers to any such software library or work
		which has been distributed under these terms.  A “work based on the
		Library” means either the Library or any derivative work under
		copyright law: that is to say, a work containing the Library or a
		portion of it, either verbatim or with modifications and/or translated
		straightforwardly into another language.  (Hereinafter, translation is
		included without limitation in the term “modification”.)

		“Source code” for a work means the preferred form of the work for
		making modifications to it.  For a library, complete source code means
		all the source code for all modules it contains, plus any associated
		interface definition files, plus the scripts used to control compilation
		and installation of the library.

		Activities other than copying, distribution and modification are not
		covered by this License; they are outside its scope.  The act of
		running a program using the Library is not restricted, and output from
		such a program is covered only if its contents constitute a work based
		on the Library (independent of the use of the Library in a tool for
		writing it).  Whether that is true depends on what the Library does
		and what the program that uses the Library does.

		**1.** You may copy and distribute verbatim copies of the Library's
		complete source code as you receive it, in any medium, provided that
		you conspicuously and appropriately publish on each copy an
		appropriate copyright notice and disclaimer of warranty; keep intact
		all the notices that refer to this License and to the absence of any
		warranty; and distribute a copy of this License along with the
		Library.

		You may charge a fee for the physical act of transferring a copy,
		and you may at your option offer warranty protection in exchange for a
		fee.

		**2.** You may modify your copy or copies of the Library or any portion
		of it, thus forming a work based on the Library, and copy and
		distribute such modifications or work under the terms of Section 1
		above, provided that you also meet all of these conditions:

		* **a)** The modified work must itself be a software library.
		* **b)** You must cause the files modified to carry prominent notices
		stating that you changed the files and the date of any change.
		* **c)** You must cause the whole of the work to be licensed at no
		charge to all third parties under the terms of this License.
		* **d)** If a facility in the modified Library refers to a function or a
		table of data to be supplied by an application program that uses
		the facility, other than as an argument passed when the facility
		is invoked, then you must make a good faith effort to ensure that,
		in the event an application does not supply such function or
		table, the facility still operates, and performs whatever part of
		its purpose remains meaningful.  
		(For example, a function in a library to compute square roots has
		a purpose that is entirely well-defined independent of the
		application.  Therefore, Subsection 2d requires that any
		application-supplied function or table used by this function must
		be optional: if the application does not supply it, the square
		root function must still compute square roots.)

		These requirements apply to the modified work as a whole.  If
		identifiable sections of that work are not derived from the Library,
		and can be reasonably considered independent and separate works in
		themselves, then this License, and its terms, do not apply to those
		sections when you distribute them as separate works.  But when you
		distribute the same sections as part of a whole which is a work based
		on the Library, the distribution of the whole must be on the terms of
		this License, whose permissions for other licensees extend to the
		entire whole, and thus to each and every part regardless of who wrote
		it.

		Thus, it is not the intent of this section to claim rights or contest
		your rights to work written entirely by you; rather, the intent is to
		exercise the right to control the distribution of derivative or
		collective works based on the Library.

		In addition, mere aggregation of another work not based on the Library
		with the Library (or with a work based on the Library) on a volume of
		a storage or distribution medium does not bring the other work under
		the scope of this License.

		**3.** You may opt to apply the terms of the ordinary GNU General Public
		License instead of this License to a given copy of the Library.  To do
		this, you must alter all the notices that refer to this License, so
		that they refer to the ordinary GNU General Public License, version 2,
		instead of to this License.  (If a newer version than version 2 of the
		ordinary GNU General Public License has appeared, then you can specify
		that version instead if you wish.)  Do not make any other change in
		these notices.

		Once this change is made in a given copy, it is irreversible for
		that copy, so the ordinary GNU General Public License applies to all
		subsequent copies and derivative works made from that copy.

		This option is useful when you wish to copy part of the code of
		the Library into a program that is not a library.

		**4.** You may copy and distribute the Library (or a portion or
		derivative of it, under Section 2) in object code or executable form
		under the terms of Sections 1 and 2 above provided that you accompany
		it with the complete corresponding machine-readable source code, which
		must be distributed under the terms of Sections 1 and 2 above on a
		medium customarily used for software interchange.

		If distribution of object code is made by offering access to copy
		from a designated place, then offering equivalent access to copy the
		source code from the same place satisfies the requirement to
		distribute the source code, even though third parties are not
		compelled to copy the source along with the object code.

		**5.** A program that contains no derivative of any portion of the
		Library, but is designed to work with the Library by being compiled or
		linked with it, is called a “work that uses the Library”.  Such a
		work, in isolation, is not a derivative work of the Library, and
		therefore falls outside the scope of this License.

		However, linking a “work that uses the Library” with the Library
		creates an executable that is a derivative of the Library (because it
		contains portions of the Library), rather than a “work that uses the
		library”.  The executable is therefore covered by this License.
		Section 6 states terms for distribution of such executables.

		When a “work that uses the Library” uses material from a header file
		that is part of the Library, the object code for the work may be a
		derivative work of the Library even though the source code is not.
		Whether this is true is especially significant if the work can be
		linked without the Library, or if the work is itself a library.  The
		threshold for this to be true is not precisely defined by law.

		If such an object file uses only numerical parameters, data
		structure layouts and accessors, and small macros and small inline
		functions (ten lines or less in length), then the use of the object
		file is unrestricted, regardless of whether it is legally a derivative
		work.  (Executables containing this object code plus portions of the
		Library will still fall under Section 6.)

		Otherwise, if the work is a derivative of the Library, you may
		distribute the object code for the work under the terms of Section 6.
		Any executables containing that work also fall under Section 6,
		whether or not they are linked directly with the Library itself.

		**6.** As an exception to the Sections above, you may also combine or
		link a “work that uses the Library” with the Library to produce a
		work containing portions of the Library, and distribute that work
		under terms of your choice, provided that the terms permit
		modification of the work for the customer's own use and reverse
		engineering for debugging such modifications.

		You must give prominent notice with each copy of the work that the
		Library is used in it and that the Library and its use are covered by
		this License.  You must supply a copy of this License.  If the work
		during execution displays copyright notices, you must include the
		copyright notice for the Library among them, as well as a reference
		directing the user to the copy of this License.  Also, you must do one
		of these things:

		* **a)** Accompany the work with the complete corresponding
		machine-readable source code for the Library including whatever
		changes were used in the work (which must be distributed under
		Sections 1 and 2 above); and, if the work is an executable linked
		with the Library, with the complete machine-readable “work that
		uses the Library”, as object code and/or source code, so that the
		user can modify the Library and then relink to produce a modified
		executable containing the modified Library.  (It is understood
		that the user who changes the contents of definitions files in the
		Library will not necessarily be able to recompile the application
		to use the modified definitions.)
		* **b)** Use a suitable shared library mechanism for linking with the
		Library.  A suitable mechanism is one that (1) uses at run time a
		copy of the library already present on the user's computer system,
		rather than copying library functions into the executable, and (2)
		will operate properly with a modified version of the library, if
		the user installs one, as long as the modified version is
		interface-compatible with the version that the work was made with.
		* **c)** Accompany the work with a written offer, valid for at
		least three years, to give the same user the materials
		specified in Subsection 6a, above, for a charge no more
		than the cost of performing this distribution.
		* **d)** If distribution of the work is made by offering access to copy
		from a designated place, offer equivalent access to copy the above
		specified materials from the same place.
		* **e)** Verify that the user has already received a copy of these
		materials or that you have already sent this user a copy.

		For an executable, the required form of the “work that uses the
		Library” must include any data and utility programs needed for
		reproducing the executable from it.  However, as a special exception,
		the materials to be distributed need not include anything that is
		normally distributed (in either source or binary form) with the major
		components (compiler, kernel, and so on) of the operating system on
		which the executable runs, unless that component itself accompanies
		the executable.

		It may happen that this requirement contradicts the license
		restrictions of other proprietary libraries that do not normally
		accompany the operating system.  Such a contradiction means you cannot
		use both them and the Library together in an executable that you
		distribute.

		**7.** You may place library facilities that are a work based on the
		Library side-by-side in a single library together with other library
		facilities not covered by this License, and distribute such a combined
		library, provided that the separate distribution of the work based on
		the Library and of the other library facilities is otherwise
		permitted, and provided that you do these two things:

		* **a)** Accompany the combined library with a copy of the same work
		based on the Library, uncombined with any other library
		facilities.  This must be distributed under the terms of the
		Sections above.
		* **b)** Give prominent notice with the combined library of the fact
		that part of it is a work based on the Library, and explaining
		where to find the accompanying uncombined form of the same work.

		**8.** You may not copy, modify, sublicense, link with, or distribute
		the Library except as expressly provided under this License.  Any
		attempt otherwise to copy, modify, sublicense, link with, or
		distribute the Library is void, and will automatically terminate your
		rights under this License.  However, parties who have received copies,
		or rights, from you under this License will not have their licenses
		terminated so long as such parties remain in full compliance.

		**9.** You are not required to accept this License, since you have not
		signed it.  However, nothing else grants you permission to modify or
		distribute the Library or its derivative works.  These actions are
		prohibited by law if you do not accept this License.  Therefore, by
		modifying or distributing the Library (or any work based on the
		Library), you indicate your acceptance of this License to do so, and
		all its terms and conditions for copying, distributing or modifying
		the Library or works based on it.

		**10.** Each time you redistribute the Library (or any work based on the
		Library), the recipient automatically receives a license from the
		original licensor to copy, distribute, link with or modify the Library
		subject to these terms and conditions.  You may not impose any further
		restrictions on the recipients' exercise of the rights granted herein.
		You are not responsible for enforcing compliance by third parties with
		this License.

		**11.** If, as a consequence of a court judgment or allegation of patent
		infringement or for any other reason (not limited to patent issues),
		conditions are imposed on you (whether by court order, agreement or
		otherwise) that contradict the conditions of this License, they do not
		excuse you from the conditions of this License.  If you cannot
		distribute so as to satisfy simultaneously your obligations under this
		License and any other pertinent obligations, then as a consequence you
		may not distribute the Library at all.  For example, if a patent
		license would not permit royalty-free redistribution of the Library by
		all those who receive copies directly or indirectly through you, then
		the only way you could satisfy both it and this License would be to
		refrain entirely from distribution of the Library.

		If any portion of this section is held invalid or unenforceable under any
		particular circumstance, the balance of the section is intended to apply,
		and the section as a whole is intended to apply in other circumstances.

		It is not the purpose of this section to induce you to infringe any
		patents or other property right claims or to contest validity of any
		such claims; this section has the sole purpose of protecting the
		integrity of the free software distribution system which is
		implemented by public license practices.  Many people have made
		generous contributions to the wide range of software distributed
		through that system in reliance on consistent application of that
		system; it is up to the author/donor to decide if he or she is willing
		to distribute software through any other system and a licensee cannot
		impose that choice.

		This section is intended to make thoroughly clear what is believed to
		be a consequence of the rest of this License.

		**12.** If the distribution and/or use of the Library is restricted in
		certain countries either by patents or by copyrighted interfaces, the
		original copyright holder who places the Library under this License may add
		an explicit geographical distribution limitation excluding those countries,
		so that distribution is permitted only in or among countries not thus
		excluded.  In such case, this License incorporates the limitation as if
		written in the body of this License.

		**13.** The Free Software Foundation may publish revised and/or new
		versions of the Lesser General Public License from time to time.
		Such new versions will be similar in spirit to the present version,
		but may differ in detail to address new problems or concerns.

		Each version is given a distinguishing version number.  If the Library
		specifies a version number of this License which applies to it and
		“any later version”, you have the option of following the terms and
		conditions either of that version or of any later version published by
		the Free Software Foundation.  If the Library does not specify a
		license version number, you may choose any version ever published by
		the Free Software Foundation.

		**14.** If you wish to incorporate parts of the Library into other free
		programs whose distribution conditions are incompatible with these,
		write to the author to ask for permission.  For software which is
		copyrighted by the Free Software Foundation, write to the Free
		Software Foundation; we sometimes make exceptions for this.  Our
		decision will be guided by the two goals of preserving the free status
		of all derivatives of our free software and of promoting the sharing
		and reuse of software generally.

	NO WARRANTY:

		**15.** BECAUSE THE LIBRARY IS LICENSED FREE OF CHARGE, THERE IS NO
		WARRANTY FOR THE LIBRARY, TO THE EXTENT PERMITTED BY APPLICABLE LAW.
		EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR
		OTHER PARTIES PROVIDE THE LIBRARY “AS IS” WITHOUT WARRANTY OF ANY
		KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
		IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
		PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE
		LIBRARY IS WITH YOU.  SHOULD THE LIBRARY PROVE DEFECTIVE, YOU ASSUME
		THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

		**16.** IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN
		WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY
		AND/OR REDISTRIBUTE THE LIBRARY AS PERMITTED ABOVE, BE LIABLE TO YOU
		FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR
		CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE
		LIBRARY (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING
		RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A
		FAILURE OF THE LIBRARY TO OPERATE WITH ANY OTHER SOFTWARE), EVEN IF
		SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
		DAMAGES.

		_END OF TERMS AND CONDITIONS_

	How to Apply These Terms to Your New Libraries:

		If you develop a new library, and you want it to be of the greatest
		possible use to the public, we recommend making it free software that
		everyone can redistribute and change.  You can do so by permitting
		redistribution under these terms (or, alternatively, under the terms of the
		ordinary General Public License).

		To apply these terms, attach the following notices to the library.  It is
		safest to attach them to the start of each source file to most effectively
		convey the exclusion of warranty; and each file should have at least the
		“copyright” line and a pointer to where the full notice is found.

		    <one line to give the library's name and a brief idea of what it does.>
		    Copyright (C) <year>  <name of author>

		    This library is free software; you can redistribute it and/or
		    modify it under the terms of the GNU Lesser General Public
		    License as published by the Free Software Foundation; either
		    version 2.1 of the License, or (at your option) any later version.

		    This library is distributed in the hope that it will be useful,
		    but WITHOUT ANY WARRANTY; without even the implied warranty of
		    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
		    Lesser General Public License for more details.

		    You should have received a copy of the GNU Lesser General Public
		    License along with this library; if not, write to the Free Software
		    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

		Also add information on how to contact you by electronic and paper mail.

		You should also get your employer (if you work as a programmer) or your
		school, if any, to sign a “copyright disclaimer” for the library, if
		necessary.  Here is a sample; alter the names:

		    Yoyodyne, Inc., hereby disclaims all copyright interest in the
		    library `Frob' (a library for tweaking knobs) written by James Random Hacker.

		    <signature of Ty Coon>, 1 April 1990
		    Ty Coon, President of Vice

		That's all there is to it!

	<a id="LGPL_NVDA">NVDA controler client</a>:
		GNU Lesser General Public License
		=================================

		_Version 2.1, February 1999_  
		_Copyright © 1991, 1999 Free Software Foundation, Inc._  
		_51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA_

		Everyone is permitted to copy and distribute verbatim copies
		of this license document, but changing it is not allowed.

		_This is the first released version of the Lesser GPL.  It also counts
		as the successor of the GNU Library Public License, version 2, hence
		the version number 2.1._

	Preamble:

		The licenses for most software are designed to take away your
		freedom to share and change it.  By contrast, the GNU General Public
		Licenses are intended to guarantee your freedom to share and change
		free software--to make sure the software is free for all its users.

		This license, the Lesser General Public License, applies to some
		specially designated software packages--typically libraries--of the
		Free Software Foundation and other authors who decide to use it.  You
		can use it too, but we suggest you first think carefully about whether
		this license or the ordinary General Public License is the better
		strategy to use in any particular case, based on the explanations below.

		When we speak of free software, we are referring to freedom of use,
		not price.  Our General Public Licenses are designed to make sure that
		you have the freedom to distribute copies of free software (and charge
		for this service if you wish); that you receive source code or can get
		it if you want it; that you can change the software and use pieces of
		it in new free programs; and that you are informed that you can do
		these things.

		To protect your rights, we need to make restrictions that forbid
		distributors to deny you these rights or to ask you to surrender these
		rights.  These restrictions translate to certain responsibilities for
		you if you distribute copies of the library or if you modify it.

		For example, if you distribute copies of the library, whether gratis
		or for a fee, you must give the recipients all the rights that we gave
		you.  You must make sure that they, too, receive or can get the source
		code.  If you link other code with the library, you must provide
		complete object files to the recipients, so that they can relink them
		with the library after making changes to the library and recompiling
		it.  And you must show them these terms so they know their rights.

		We protect your rights with a two-step method: **(1)** we copyright the
		library, and **(2)** we offer you this license, which gives you legal
		permission to copy, distribute and/or modify the library.

		To protect each distributor, we want to make it very clear that
		there is no warranty for the free library.  Also, if the library is
		modified by someone else and passed on, the recipients should know
		that what they have is not the original version, so that the original
		author's reputation will not be affected by problems that might be
		introduced by others.

		Finally, software patents pose a constant threat to the existence of
		any free program.  We wish to make sure that a company cannot
		effectively restrict the users of a free program by obtaining a
		restrictive license from a patent holder.  Therefore, we insist that
		any patent license obtained for a version of the library must be
		consistent with the full freedom of use specified in this license.

		Most GNU software, including some libraries, is covered by the
		ordinary GNU General Public License.  This license, the GNU Lesser
		General Public License, applies to certain designated libraries, and
		is quite different from the ordinary General Public License.  We use
		this license for certain libraries in order to permit linking those
		libraries into non-free programs.

		When a program is linked with a library, whether statically or using
		a shared library, the combination of the two is legally speaking a
		combined work, a derivative of the original library.  The ordinary
		General Public License therefore permits such linking only if the
		entire combination fits its criteria of freedom.  The Lesser General
		Public License permits more lax criteria for linking other code with
		the library.

		We call this license the “Lesser” General Public License because it
		does Less to protect the user's freedom than the ordinary General
		Public License.  It also provides other free software developers Less
		of an advantage over competing non-free programs.  These disadvantages
		are the reason we use the ordinary General Public License for many
		libraries.  However, the Lesser license provides advantages in certain
		special circumstances.

		For example, on rare occasions, there may be a special need to
		encourage the widest possible use of a certain library, so that it becomes
		a de-facto standard.  To achieve this, non-free programs must be
		allowed to use the library.  A more frequent case is that a free
		library does the same job as widely used non-free libraries.  In this
		case, there is little to gain by limiting the free library to free
		software only, so we use the Lesser General Public License.

		In other cases, permission to use a particular library in non-free
		programs enables a greater number of people to use a large body of
		free software.  For example, permission to use the GNU C Library in
		non-free programs enables many more people to use the whole GNU
		operating system, as well as its variant, the GNU/Linux operating
		system.

		Although the Lesser General Public License is Less protective of the
		users' freedom, it does ensure that the user of a program that is
		linked with the Library has the freedom and the wherewithal to run
		that program using a modified version of the Library.

		The precise terms and conditions for copying, distribution and
		modification follow.  Pay close attention to the difference between a
		“work based on the library” and a “work that uses the library”.  The
		former contains code derived from the library, whereas the latter must
		be combined with the library in order to run.

	TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION:

		**0.** This License Agreement applies to any software library or other
		program which contains a notice placed by the copyright holder or
		other authorized party saying it may be distributed under the terms of
		this Lesser General Public License (also called “this License”).
		Each licensee is addressed as “you”.

		A “library” means a collection of software functions and/or data
		prepared so as to be conveniently linked with application programs
		(which use some of those functions and data) to form executables.

		The “Library”, below, refers to any such software library or work
		which has been distributed under these terms.  A “work based on the
		Library” means either the Library or any derivative work under
		copyright law: that is to say, a work containing the Library or a
		portion of it, either verbatim or with modifications and/or translated
		straightforwardly into another language.  (Hereinafter, translation is
		included without limitation in the term “modification”.)

		“Source code” for a work means the preferred form of the work for
		making modifications to it.  For a library, complete source code means
		all the source code for all modules it contains, plus any associated
		interface definition files, plus the scripts used to control compilation
		and installation of the library.

		Activities other than copying, distribution and modification are not
		covered by this License; they are outside its scope.  The act of
		running a program using the Library is not restricted, and output from
		such a program is covered only if its contents constitute a work based
		on the Library (independent of the use of the Library in a tool for
		writing it).  Whether that is true depends on what the Library does
		and what the program that uses the Library does.

		**1.** You may copy and distribute verbatim copies of the Library's
		complete source code as you receive it, in any medium, provided that
		you conspicuously and appropriately publish on each copy an
		appropriate copyright notice and disclaimer of warranty; keep intact
		all the notices that refer to this License and to the absence of any
		warranty; and distribute a copy of this License along with the
		Library.

		You may charge a fee for the physical act of transferring a copy,
		and you may at your option offer warranty protection in exchange for a
		fee.

		**2.** You may modify your copy or copies of the Library or any portion
		of it, thus forming a work based on the Library, and copy and
		distribute such modifications or work under the terms of Section 1
		above, provided that you also meet all of these conditions:

		* **a)** The modified work must itself be a software library.
		* **b)** You must cause the files modified to carry prominent notices
		stating that you changed the files and the date of any change.
		* **c)** You must cause the whole of the work to be licensed at no
		charge to all third parties under the terms of this License.
		* **d)** If a facility in the modified Library refers to a function or a
		table of data to be supplied by an application program that uses
		the facility, other than as an argument passed when the facility
		is invoked, then you must make a good faith effort to ensure that,
		in the event an application does not supply such function or
		table, the facility still operates, and performs whatever part of
		its purpose remains meaningful.  
		(For example, a function in a library to compute square roots has
		a purpose that is entirely well-defined independent of the
		application.  Therefore, Subsection 2d requires that any
		application-supplied function or table used by this function must
		be optional: if the application does not supply it, the square
		root function must still compute square roots.)

		These requirements apply to the modified work as a whole.  If
		identifiable sections of that work are not derived from the Library,
		and can be reasonably considered independent and separate works in
		themselves, then this License, and its terms, do not apply to those
		sections when you distribute them as separate works.  But when you
		distribute the same sections as part of a whole which is a work based
		on the Library, the distribution of the whole must be on the terms of
		this License, whose permissions for other licensees extend to the
		entire whole, and thus to each and every part regardless of who wrote
		it.

		Thus, it is not the intent of this section to claim rights or contest
		your rights to work written entirely by you; rather, the intent is to
		exercise the right to control the distribution of derivative or
		collective works based on the Library.

		In addition, mere aggregation of another work not based on the Library
		with the Library (or with a work based on the Library) on a volume of
		a storage or distribution medium does not bring the other work under
		the scope of this License.

		**3.** You may opt to apply the terms of the ordinary GNU General Public
		License instead of this License to a given copy of the Library.  To do
		this, you must alter all the notices that refer to this License, so
		that they refer to the ordinary GNU General Public License, version 2,
		instead of to this License.  (If a newer version than version 2 of the
		ordinary GNU General Public License has appeared, then you can specify
		that version instead if you wish.)  Do not make any other change in
		these notices.

		Once this change is made in a given copy, it is irreversible for
		that copy, so the ordinary GNU General Public License applies to all
		subsequent copies and derivative works made from that copy.

		This option is useful when you wish to copy part of the code of
		the Library into a program that is not a library.

		**4.** You may copy and distribute the Library (or a portion or
		derivative of it, under Section 2) in object code or executable form
		under the terms of Sections 1 and 2 above provided that you accompany
		it with the complete corresponding machine-readable source code, which
		must be distributed under the terms of Sections 1 and 2 above on a
		medium customarily used for software interchange.

		If distribution of object code is made by offering access to copy
		from a designated place, then offering equivalent access to copy the
		source code from the same place satisfies the requirement to
		distribute the source code, even though third parties are not
		compelled to copy the source along with the object code.

		**5.** A program that contains no derivative of any portion of the
		Library, but is designed to work with the Library by being compiled or
		linked with it, is called a “work that uses the Library”.  Such a
		work, in isolation, is not a derivative work of the Library, and
		therefore falls outside the scope of this License.

		However, linking a “work that uses the Library” with the Library
		creates an executable that is a derivative of the Library (because it
		contains portions of the Library), rather than a “work that uses the
		library”.  The executable is therefore covered by this License.
		Section 6 states terms for distribution of such executables.

		When a “work that uses the Library” uses material from a header file
		that is part of the Library, the object code for the work may be a
		derivative work of the Library even though the source code is not.
		Whether this is true is especially significant if the work can be
		linked without the Library, or if the work is itself a library.  The
		threshold for this to be true is not precisely defined by law.

		If such an object file uses only numerical parameters, data
		structure layouts and accessors, and small macros and small inline
		functions (ten lines or less in length), then the use of the object
		file is unrestricted, regardless of whether it is legally a derivative
		work.  (Executables containing this object code plus portions of the
		Library will still fall under Section 6.)

		Otherwise, if the work is a derivative of the Library, you may
		distribute the object code for the work under the terms of Section 6.
		Any executables containing that work also fall under Section 6,
		whether or not they are linked directly with the Library itself.

		**6.** As an exception to the Sections above, you may also combine or
		link a “work that uses the Library” with the Library to produce a
		work containing portions of the Library, and distribute that work
		under terms of your choice, provided that the terms permit
		modification of the work for the customer's own use and reverse
		engineering for debugging such modifications.

		You must give prominent notice with each copy of the work that the
		Library is used in it and that the Library and its use are covered by
		this License.  You must supply a copy of this License.  If the work
		during execution displays copyright notices, you must include the
		copyright notice for the Library among them, as well as a reference
		directing the user to the copy of this License.  Also, you must do one
		of these things:

		* **a)** Accompany the work with the complete corresponding
		machine-readable source code for the Library including whatever
		changes were used in the work (which must be distributed under
		Sections 1 and 2 above); and, if the work is an executable linked
		with the Library, with the complete machine-readable “work that
		uses the Library”, as object code and/or source code, so that the
		user can modify the Library and then relink to produce a modified
		executable containing the modified Library.  (It is understood
		that the user who changes the contents of definitions files in the
		Library will not necessarily be able to recompile the application
		to use the modified definitions.)
		* **b)** Use a suitable shared library mechanism for linking with the
		Library.  A suitable mechanism is one that (1) uses at run time a
		copy of the library already present on the user's computer system,
		rather than copying library functions into the executable, and (2)
		will operate properly with a modified version of the library, if
		the user installs one, as long as the modified version is
		interface-compatible with the version that the work was made with.
		* **c)** Accompany the work with a written offer, valid for at
		least three years, to give the same user the materials
		specified in Subsection 6a, above, for a charge no more
		than the cost of performing this distribution.
		* **d)** If distribution of the work is made by offering access to copy
		from a designated place, offer equivalent access to copy the above
		specified materials from the same place.
		* **e)** Verify that the user has already received a copy of these
		materials or that you have already sent this user a copy.

		For an executable, the required form of the “work that uses the
		Library” must include any data and utility programs needed for
		reproducing the executable from it.  However, as a special exception,
		the materials to be distributed need not include anything that is
		normally distributed (in either source or binary form) with the major
		components (compiler, kernel, and so on) of the operating system on
		which the executable runs, unless that component itself accompanies
		the executable.

		It may happen that this requirement contradicts the license
		restrictions of other proprietary libraries that do not normally
		accompany the operating system.  Such a contradiction means you cannot
		use both them and the Library together in an executable that you
		distribute.

		**7.** You may place library facilities that are a work based on the
		Library side-by-side in a single library together with other library
		facilities not covered by this License, and distribute such a combined
		library, provided that the separate distribution of the work based on
		the Library and of the other library facilities is otherwise
		permitted, and provided that you do these two things:

		* **a)** Accompany the combined library with a copy of the same work
		based on the Library, uncombined with any other library
		facilities.  This must be distributed under the terms of the
		Sections above.
		* **b)** Give prominent notice with the combined library of the fact
		that part of it is a work based on the Library, and explaining
		where to find the accompanying uncombined form of the same work.

		**8.** You may not copy, modify, sublicense, link with, or distribute
		the Library except as expressly provided under this License.  Any
		attempt otherwise to copy, modify, sublicense, link with, or
		distribute the Library is void, and will automatically terminate your
		rights under this License.  However, parties who have received copies,
		or rights, from you under this License will not have their licenses
		terminated so long as such parties remain in full compliance.

		**9.** You are not required to accept this License, since you have not
		signed it.  However, nothing else grants you permission to modify or
		distribute the Library or its derivative works.  These actions are
		prohibited by law if you do not accept this License.  Therefore, by
		modifying or distributing the Library (or any work based on the
		Library), you indicate your acceptance of this License to do so, and
		all its terms and conditions for copying, distributing or modifying
		the Library or works based on it.

		**10.** Each time you redistribute the Library (or any work based on the
		Library), the recipient automatically receives a license from the
		original licensor to copy, distribute, link with or modify the Library
		subject to these terms and conditions.  You may not impose any further
		restrictions on the recipients' exercise of the rights granted herein.
		You are not responsible for enforcing compliance by third parties with
		this License.

		**11.** If, as a consequence of a court judgment or allegation of patent
		infringement or for any other reason (not limited to patent issues),
		conditions are imposed on you (whether by court order, agreement or
		otherwise) that contradict the conditions of this License, they do not
		excuse you from the conditions of this License.  If you cannot
		distribute so as to satisfy simultaneously your obligations under this
		License and any other pertinent obligations, then as a consequence you
		may not distribute the Library at all.  For example, if a patent
		license would not permit royalty-free redistribution of the Library by
		all those who receive copies directly or indirectly through you, then
		the only way you could satisfy both it and this License would be to
		refrain entirely from distribution of the Library.

		If any portion of this section is held invalid or unenforceable under any
		particular circumstance, the balance of the section is intended to apply,
		and the section as a whole is intended to apply in other circumstances.

		It is not the purpose of this section to induce you to infringe any
		patents or other property right claims or to contest validity of any
		such claims; this section has the sole purpose of protecting the
		integrity of the free software distribution system which is
		implemented by public license practices.  Many people have made
		generous contributions to the wide range of software distributed
		through that system in reliance on consistent application of that
		system; it is up to the author/donor to decide if he or she is willing
		to distribute software through any other system and a licensee cannot
		impose that choice.

		This section is intended to make thoroughly clear what is believed to
		be a consequence of the rest of this License.

		**12.** If the distribution and/or use of the Library is restricted in
		certain countries either by patents or by copyrighted interfaces, the
		original copyright holder who places the Library under this License may add
		an explicit geographical distribution limitation excluding those countries,
		so that distribution is permitted only in or among countries not thus
		excluded.  In such case, this License incorporates the limitation as if
		written in the body of this License.

		**13.** The Free Software Foundation may publish revised and/or new
		versions of the Lesser General Public License from time to time.
		Such new versions will be similar in spirit to the present version,
		but may differ in detail to address new problems or concerns.

		Each version is given a distinguishing version number.  If the Library
		specifies a version number of this License which applies to it and
		“any later version”, you have the option of following the terms and
		conditions either of that version or of any later version published by
		the Free Software Foundation.  If the Library does not specify a
		license version number, you may choose any version ever published by
		the Free Software Foundation.

		**14.** If you wish to incorporate parts of the Library into other free
		programs whose distribution conditions are incompatible with these,
		write to the author to ask for permission.  For software which is
		copyrighted by the Free Software Foundation, write to the Free
		Software Foundation; we sometimes make exceptions for this.  Our
		decision will be guided by the two goals of preserving the free status
		of all derivatives of our free software and of promoting the sharing
		and reuse of software generally.

	NO WARRANTY:

		**15.** BECAUSE THE LIBRARY IS LICENSED FREE OF CHARGE, THERE IS NO
		WARRANTY FOR THE LIBRARY, TO THE EXTENT PERMITTED BY APPLICABLE LAW.
		EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR
		OTHER PARTIES PROVIDE THE LIBRARY “AS IS” WITHOUT WARRANTY OF ANY
		KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
		IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
		PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE
		LIBRARY IS WITH YOU.  SHOULD THE LIBRARY PROVE DEFECTIVE, YOU ASSUME
		THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

		**16.** IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN
		WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY
		AND/OR REDISTRIBUTE THE LIBRARY AS PERMITTED ABOVE, BE LIABLE TO YOU
		FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR
		CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE
		LIBRARY (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING
		RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A
		FAILURE OF THE LIBRARY TO OPERATE WITH ANY OTHER SOFTWARE), EVEN IF
		SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
		DAMAGES.

		_END OF TERMS AND CONDITIONS_

	How to Apply These Terms to Your New Libraries:

		If you develop a new library, and you want it to be of the greatest
		possible use to the public, we recommend making it free software that
		everyone can redistribute and change.  You can do so by permitting
		redistribution under these terms (or, alternatively, under the terms of the
		ordinary General Public License).

		To apply these terms, attach the following notices to the library.  It is
		safest to attach them to the start of each source file to most effectively
		convey the exclusion of warranty; and each file should have at least the
		“copyright” line and a pointer to where the full notice is found.

		    <one line to give the library's name and a brief idea of what it does.>
		    Copyright (C) <year>  <name of author>

		    This library is free software; you can redistribute it and/or
		    modify it under the terms of the GNU Lesser General Public
		    License as published by the Free Software Foundation; either
		    version 2.1 of the License, or (at your option) any later version.

		    This library is distributed in the hope that it will be useful,
		    but WITHOUT ANY WARRANTY; without even the implied warranty of
		    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
		    Lesser General Public License for more details.

		    You should have received a copy of the GNU Lesser General Public
		    License along with this library; if not, write to the Free Software
		    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

		Also add information on how to contact you by electronic and paper mail.

		You should also get your employer (if you work as a programmer) or your
		school, if any, to sign a “copyright disclaimer” for the library, if
		necessary.  Here is a sample; alter the names:

		    Yoyodyne, Inc., hereby disclaims all copyright interest in the
		    library `Frob' (a library for tweaking knobs) written by James Random Hacker.

		    <signature of Ty Coon>, 1 April 1990
		    Ty Coon, President of Vice

		That's all there is to it!

	MIT licensed code:

	<a id="MIT_cmp">cmp - msgpack serialization library</a>:

		Copyright (c) 2020 Charles Gunyon

		Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

		The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	<a id="MIT_enet">enet library</a>:

		Copyright (c) 2002-2016 Lee Salzman

		Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

		The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	<a id="MIT_hammerspoon">Hammerspoon's input box dialog extension for MacOS</a>:

		Copyright (c) 2017 Chris Hocking

		Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

		The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	<a id="MIT_ratas">Ratas - hierarchical timer wheel</a>:

		Copyright (c) 2016 Juho Snellman

		Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

		The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	<a id="MIT_universal_speech">UniversalSpeech - screen reader library for windows</a>:

		Copyright (c) 2011-2018 Quentin cosendey

		Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

		The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	<a id="MIT_unordered_dense">unordered_dense - fast unordered map and sets for c++</a>:

		Copyright (c) 2022-2023 Martin Leitner-Ankerl <martin.ankerl@gmail.com>

		Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

		The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	<a id="MIT_zodiac">Zodiac's angelscript print function</a>:
		Copyright (c) 2021 SpehleonLP

		Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

		The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	BSD 2 or 3 clause licensed code:

	<a id="BSD_bsd">Code from the FreeBSD Project</a>:

		Copyright (c) 1983, 1993
		The Regents of the University of California.  All rights reserved.

		Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

		1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
		2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
		3. Neither the name of the University nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

		THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	<a id="BSD_double_conversion">double-conversion</a>:
		Copyright 2010 the V8 project authors. All rights reserved.

		Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

		* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

		* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

		* Neither the name of Google Inc. nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	<a id="BSD_opus">libopus - opus audio codec</a>:

		Copyright (c) 2010-2011 Xiph.Org Foundation, Skype Limited
		   Written by Jean-Marc Valin and Koen Vos

		Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

		* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

		* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	<a id="BSD_PCRE">PCRE - Perl Compatible Regular Expressions</a>:

		Copyright (c) 1997-2014 University of Cambridge
		All rights reserved.

		Written by: Philip Hazel <ph10@cam.ac.uk>

		University of Cambridge Computing Service,
		Cambridge, England. Phone: +44 1223 334714.

		Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

		* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

		* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

		* Neither the name of the University of Cambridge nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	EXEMPTION FOR BINARY LIBRARY-LIKE PACKAGES:
		The second condition in the BSD licence (covering binary redistributions) does not apply all the way down a chain of software. If binary package A includes PCRE2, it must respect the condition, but if package B is software that includes package A, the condition is not imposed on package B unless it uses PCRE2 independently.


	<a id="BSD_vorbis">libvorbis - vorbis audio codec</a>:

		Copyright (c) 2002-2020 Xiph.org Foundation

		Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

		* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

		* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

		* Neither the name of the Xiph.org Foundation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

	<a id="BSD_wepoll">wepoll - epoll support for Windows</a>:
		Copyright 2012-2020, Bert Belder <bertbelder@gmail.com>

		All rights reserved.

		Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

		* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

		* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



