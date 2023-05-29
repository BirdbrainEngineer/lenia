# Birdbrain's Lenia renderer
This repository contains my quick and dirty "workspace" to explore the [Standard Lenia](https://arxiv.org/abs/1812.05433) and [Expanded Lenia](https://arxiv.org/abs/2005.03742) cellular automata systems. The code is written purely in Rust and uses [lenia_ca](https://github.com/BirdbrainEngineer/lenia_ca) crate by your's truly to simulate the systems. Since the ``lenia_ca`` crate makes heavy use of [ndarray](https://docs.rs/ndarray/latest/ndarray/) crate, then you should also familiarize yourself with that.

It is recommended that instead of using this Lenia renderer, you make your own, as this renderer is extremely unoptimized and rather slow. You can simply use the ``lenia_ca`` crate for the backend simulation.

I also created a video, which can be found on [YouTube](https://www.youtube.com/channel/UCZDOT6k11nLH3ZwA6Xp89NA), which displays plenty of animations made with this workspace. 

Should you still choose to use and compile this renderer then here are some important bits of information:

* You should compile by running `cargo build --release`, and also run the release version with `cargo run --release`, or do both of them together with `cargo build --release && cargo run --release`
* Depending on how "beefy" your cpu is, you may have to reduce `const X_SIDE_LEN` and `const Y_SIDE_LEN`. My system has a Ryzen 7 7700X and I could run 720p resolution at 60fps, and 1080p resolution at around 30 to 40, as long as the number of channels and kernels did not exceed the number of physical cores (8 in my case). 
* If you wish to use this code to simulate your own Lenia systems (rather than the auto-generated ones), then the easiest way to do so, is to set `rules: Vec<Vec<f64>>` equal to `Vec::new()`, and then change the kernels, growth functions, weights, etc... of the `lenia_simulator` variable. It is important that you do not press "n", "m" nor "," during the operation of the simulation in that case though, as it would crash the program :)
* Don't get discouraged if you can't find interesting looking "rulesets" - The interesting stuff happens at a, relatively speaking, narrow ranges of parameters... it needs just the correct amount of stability, but also still be chaotic enough to make the interesting dynamics happen.

Important key bindings (you can change them by modifying them in the `match keyboardstate.character` match block).
* "k" - toggles between viewing the kernels or simulation
* "r" - randomly seeds the simulation board based on constants earlier in the code
* "s" - toggles continuous simulating
* "i" - performs a single iteration of the simulation

If using the code unchanged then the following are also important
* "n" - Changes the currently used rulesets completely
* "m" - Uses the currently set ruleset as basis and tweaks the ruleset slightly for a slightly different result
* "," - Permanently tweaks the rulesets slightly from the currently used ruleset

![Example screenshots](demoscreenshots.png)
1. Orbium unicaudatus - The iconic Lenia glider
2. Tricircium inversus - An oscillator with 3-fold symmetry
3. Astrium inversus - An oscillator with 5-fold symmetry
4. A generic glider arising from a 2-channel & 2-kernel interaction
5. Lots of multi-channel & multi-kernel interacting "single cell organisms"
6. An asymmetric glider arising from complex set of channels and kernels.
7. "Snakes and Parasites"
8. Tetrahedrome rotans - A 3D rotating oscillator with interesting symmetries (rendered with blender)
9. A 3D slice of a chaotic oscillator in 4D (rendered with Blender)