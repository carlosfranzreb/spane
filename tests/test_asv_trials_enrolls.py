import os
import json
import unittest
import shutil

from spkanon_eval.evaluation.asv.trials_enrolls import split_trials_enrolls


class TestTrialsEnrolls(unittest.TestCase):
    def setUp(self):

        # create/empty experiment folder
        self.exp_folder = "spkanon_eval/tests/logs/split_trials_enrolls"
        if os.path.isdir(self.exp_folder):
            shutil.rmtree(self.exp_folder)
        os.makedirs(os.path.join(self.exp_folder, "data"))

        # add the original datafile to the experiment folder
        self.root_folder = "tests/data"
        datafile = "spkanon_eval/tests/datafiles/ls-dev-clean-2.txt"
        shutil.copy(datafile, os.path.join(self.exp_folder, "data", "eval.txt"))

        # create an anonymized datafile in the experiment folder
        writer = open(os.path.join(self.exp_folder, "data", "anon_eval.txt"), "w")
        anon_folder = os.path.join(self.exp_folder, "results", "eval")
        for line in open(datafile):
            obj = json.loads(line)
            obj["path"] = obj["path"].replace(self.root_folder, anon_folder)
            writer.write(json.dumps(obj) + "\n")

        # create trials and enrolls files
        self.expected_split = dict()
        self.expected_split["trials"] = [
            "2412-153948-0000",
            "2412-153948-0001",
            "3752-4944-0001",
            "1988-24833-0002",
        ]
        self.expected_split["enrolls"] = [
            "2412-153948-0002",
            "3752-4944-0000",
            "3752-4944-0002",
            "1988-24833-0000",
        ]

        self.split_files = dict()
        for split in ["trials", "enrolls"]:
            dump_file = os.path.join(self.exp_folder, "data", f"true_{split}.txt")
            self.split_files[split] = [dump_file]
            with open(dump_file, "w") as f:
                for fname in self.expected_split[split]:
                    f.write(fname + "\n")

    def tearDown(self):
        """Remove the created directory"""
        shutil.rmtree(self.exp_folder)

    def test_split_both_passed(self):
        """
        Test that it works when both splits are passed (trials and enrolls).
        """

        # anon folder is the same as root_folder and therefore not a kwarg
        out_files = dict()
        out_files["trials"], out_files["enrolls"] = split_trials_enrolls(
            self.exp_folder,
            True,
            root_folder=self.root_folder,
            trials=self.split_files["trials"],
            enrolls=self.split_files["enrolls"],
        )

        for split, out_file in out_files.items():
            self.assertTrue(os.path.isfile(out_file), split)

            objects, fpaths = list(), list()
            for line in open(out_file):
                obj = json.loads(line)
                objects.append(obj)
                fpaths.append(os.path.splitext(os.path.basename(obj["path"]))[0])

            # Check the content of the files
            self.assertCountEqual(fpaths, self.expected_split[split], split)

            # check that the utterances are ordered by duration
            last_dur = float("inf")
            for obj in objects:
                self.assertTrue(obj["duration"] < last_dur, split)
                last_dur = obj["duration"]


if __name__ == "__main__":
    unittest.main()
