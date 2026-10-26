(Website Link)[https://civitai.red/models/1331682/wan-2221-pov-missionary?modelVersionId=2098405]
Wan 2.2:

Download both the highnoise and lownoise loras, and plug them both into your workflow. The "training images" download contains the example I used for the previews.

T2V notes: I did not use the self forcing lora for the highnoise part of the workflow, because I find that it limits facial variety too much. In my testing, this happened with or without the missionary lora plugged in, so if you're having trouble with facial variety, you may have to disable the speed lora for the highnoise step.

I2V notes: This version should handle cases where the penis starts outside the vagina. It can handle cases where there's no man at all in the starting image, however, don't expect the penis to look very good in these cases. Wan still doesn't have a good idea of what they're supposed to look like. As with the T2V version, I found that using the speed loras for the highnoise step causes the results to be less good. For 5 seconds of video, the lightning lora seemed to cause the videos to be in slow motion when using it on the highnoise step. Maybe there are good settings that I just haven't found yet.

Wan2.1:

I2V version: While the T2V model can work really well as I2V, it seems to be picky about resolutions and aspect ratios and things like that. The specifically trained I2V model just seems to work better in more situations. So, it's becoming clear to me that providing an I2V model along with the T2V model is worth it. It uses the same training data as the T2V one, minus the static images.

Version 1.1 Updates: Trained at a higher resolution in an attempt to get higher quality outputs. I also added more training data so there's more movement and bouncing breasts.

Important parts of the prompt:

with her legs spread having sex with a man
...
A man is thrusting his penis back and forth inside her vagina at the bottom of the screen

{Movement is fast with bouncing breasts|Movement is slow}

Her breasts are {small|medium sized|large}

Sometimes it may help to put the things you don't want in the negatives. For example, if you want small breasts, putting large breasts in the negative could help.