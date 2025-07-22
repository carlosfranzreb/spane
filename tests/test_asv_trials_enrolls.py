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

        writer.close()

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

    # def tearDown(self):
    #     """Remove the created directory"""
    #     shutil.rmtree(self.exp_folder)

    def test_when_passed(self):
        """
        Test that it works when one or both splits are passed (trials, enrolls).
        """

        # anon folder is the same as root_folder and therefore not a kwarg
        passed_splits_arr = [
            {
                "name": "both_passed",
                "trials": self.split_files["trials"],
                "enrolls": self.split_files["enrolls"],
            },
            {
                "name": "trials_passed",
                "trials": self.split_files["trials"],
                "enrolls": None,
            },
            {
                "name": "enrolls_passed",
                "trials": None,
                "enrolls": self.split_files["enrolls"],
            },
        ]
        for passed_splits in passed_splits_arr:
            self.split_with_args(passed_splits)

            # delete created split files
            for split in ["trials", "enrolls"]:
                os.remove(os.path.join(self.exp_folder, "data", f"eval_{split}.txt"))

    def split_with_args(self, passed_splits: dict[str, str]):
        out_files = dict()
        out_files["trials"], out_files["enrolls"] = split_trials_enrolls(
            self.exp_folder,
            True,
            root_folder=self.root_folder,
            trials=passed_splits["trials"],
            enrolls=passed_splits["enrolls"],
        )

        for split, out_file in out_files.items():
            error_msg = f"{passed_splits['name']}-{split}"
            self.assertTrue(os.path.isfile(out_file), error_msg)

            # gather output and check that the paths are anonymized
            objects, fnames = list(), list()
            for line in open(out_file):
                obj = json.loads(line)
                objects.append(obj)
                fnames.append(os.path.splitext(os.path.basename(obj["path"]))[0])
                self.assertTrue(self.exp_folder in obj["path"], error_msg)

            # Check the content of the files
            if passed_splits[split] is not None:
                fnames_expected = self.expected_split[split]
            else:
                # get all samples except those in the other split
                split_other = "trials" if split == "enrolls" else "enrolls"
                fnames_other = self.expected_split[split_other]
                fnames_expected = list()
                for line in open(os.path.join(self.exp_folder, "data", "eval.txt")):
                    obj = json.loads(line)
                    fname = os.path.splitext(os.path.basename(obj["path"]))[0]
                    if fname not in fnames_other:
                        fnames_expected.append(fname)

            # check the content of the files
            self.assertCountEqual(fnames, fnames_expected, error_msg)

            # check that the utterances are ordered by duration
            last_dur = float("inf")
            for obj in objects:
                self.assertTrue(obj["duration"] < last_dur, error_msg)
                last_dur = obj["duration"]
