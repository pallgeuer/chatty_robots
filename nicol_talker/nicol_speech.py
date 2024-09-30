#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path


from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

import playsound

class NICOL_TALKER():

    def __init__(self, model_path='./tts_models--en--vctk--vits/model_file.pth', config_path='./tts_models--en--vctk--vits/config.json', use_cuda=True):

        self.model_path = model_path
        self.config_path = config_path

        # load models
        self.synthesizer = Synthesizer(
            self.model_path,
            self.config_path,
            None,
            None,
            None,
            None,
            None,
            None,
            use_cuda,
        )

        print("These are the available speakers " + str(self.synthesizer.tts_model.speaker_manager.ids))

        if self.synthesizer.tts_model.language_manager is not None:
            print("These are the available languages " + str(self.synthesizer.tts_model.language_manager.ids))


    def synth(self,text_to_syn,out_filename,speaker_idx="p287"):

        # RUN THE SYNTHESIS
        if text_to_syn:
            print(" > Text: {}".format(text_to_syn))

        wav = self.synthesizer.tts(
            text_to_syn,
            speaker_idx,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        # save the results
        print(" > Saving output to {}".format(out_filename))
        self.synthesizer.save_wav(wav, out_filename)
    
    def play(self,out_filename,block=False):
        playsound.playsound(out_filename,block=block)


if __name__ == "__main__":
    nicol_talker = NICOL_TALKER()
    nicol_talker.synth("Simple is better than complex. Complex is better than complicated.","./temp.wav")
    nicol_talker.play("./temp.wav")

