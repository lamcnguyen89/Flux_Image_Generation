Krea 2 RAW is the open-weights, undistilled checkpoint of Krea 2 - Krea AI's first foundation image model, trained from scratch with a focus on how an image feels rather than just what it contains. RAW is the full-guidance build with no step or CFG distillation, which gives it the highest quality ceiling of the family and makes it the right base for fine-tuning and LoRA training.

Originally released by Krea AI on Hugging Face. All credit for the model goes to the Krea team and the contributors listed below. Civitai is hosting a mirror so creators can generate with it and train on it on-site - please head to the original repo for weights, the technical report, and to support the project directly.
Built by

Krea AI, with Sangwu Lee, Erwann Millon, Le Zhuo, Matthew Newton, Andrei Filatov, Abhinay Devarinti, Dazhi Zhong, Avram Djordjevic, Gabriel Menezes, Will Beddow, Titus Ebbecke, Mihai Petrescu, Owen Fahey, Gian Saß, Felix Gil, and Victor Perez.
Undistilled, full-guidance build

RAW runs with classifier-free guidance and no distillation, so it keeps the full character of the base model. Krea's recommended starting point is around 52 steps at CFG 3.5. It generates at up to 1024px and trades speed for the cleanest, most controllable output of the Krea 2 family. Reach for RAW when you want the maximum quality ceiling or a predictable base for training.
Built for fine-tuning and LoRAs

Because it is undistilled, RAW is the intended target for post-training, fine-tunes, and LoRA training. LoRAs trained on RAW also carry over to Krea 2 Turbo, so you can train once on the full model and run inference fast on the distilled build.
On-site generation

You can generate with Krea 2 RAW directly on Civitai and stack community LoRAs on top. Weights are available upstream on Hugging Face under Krea's community license with permissive use; commercial licensing is available through opensource@krea.ai.
Links

    Weights: huggingface.co/krea/krea-2-raw
    Technical report: krea.ai/blog/krea-2-technical-report
    Product: krea.ai
