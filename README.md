<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">whisper-prep</h3>

  <p align="center">
    Data preparation utility for the finetuning of OpenAI's Whisper model.
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
This package helps you to create training data for fine-tuning whisper from sentences. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Guide -->
## Data Preparation Guide
You can run `python main.py --config train.yaml" in the base path after installing this package to see how it works.

1. **Data File (.tsv):**
   - Create a `.tsv` file with two required columns:
     - `path`: The relative path to the `.mp3` file.
     - `sentence`: The text corresponding to the audio file.
   - Optional: If a `client_id` is included, it can be used to increase the probability that following sentences are from the same speaker. Refer to `generate_fold` in `src/whisper_prep/generation/generate.py` for additional features.

2. **Configuration File (.yaml):**
   - Set up a `.yaml` configuration file. An example can be found at `tests/assets/configs/test.yaml`.

3. **Running the Generation Script:**
   - Use the `generate_fold_from_yaml` function, inputting your `.yaml` file as a dictionary, to start generating data folds.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Vincenzo Timmel - vincenzo.timmel@fhnw.ch

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-shield]: https://img.shields.io/github/issues/i4Ds/whisper-prep.svg?style=for-the-badge
[issues-url]: https://github.com/i4Ds/whisper-prep/issues
[license-shield]: https://img.shields.io/github/license/i4Ds/whisper-prep.svg?style=for-the-badge
[license-url]: https://github.com/i4Ds/whisper-prep/blob/main/LICENSE