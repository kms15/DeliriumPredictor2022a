#!/usr/bin/python3
import re
import statistics
from datetime import datetime, timedelta
import numpy as np

if __name__ == '__main__':
    data_dir = 'ImportedData/2019-01-17/'
    demeographics_filename = data_dir + 'Patients.csv'
    holdout_filename = 'Cache/2019-01-17/holdout_and_cvgroups.csv'
    cam_filename = 'ImportedData/2019-01-17/CamScores.csv'

    with open(demeographics_filename, 'r') as demographics_file, \
            open(holdout_filename, 'r') as holdout_file, \
            open(cam_filename, 'r') as cam_file:

        class StatsTally:
            def __init__(self):
                self.count = 0
                self.count_male = 0
                self.count_female = 0
                self.cam_counts = []
                self.cam_positive_fraction = []
                self.cam_day_counts = []
                self.cam_positive_day_fraction = []
                self.mean_ages = []

            def update(self, gender, cam_scores, ages, cam_days):
                self.count += 1
                if gender == 'Male':
                    self.count_male += 1
                elif gender == 'Female':
                    self.count_female += 1
                self.cam_counts.append(len(cam_scores))
                self.cam_positive_fraction.append(sum(cam_scores)/len(cam_scores))
                self.cam_day_counts.append(len(cam_days))
                self.cam_positive_day_fraction.append(sum(cam_days)/len(cam_days))
                self.mean_ages.append(statistics.mean(ages))

        all_tally = StatsTally()
        train_tally = StatsTally()
        holdout_tally = StatsTally()
        cam_positive_tally = StatsTally()
        cam_negative_tally = StatsTally()

        mrn_cam = 0

        # skip the headers
        demographics_file.readline()
        holdout_file.readline()
        cam_file.readline()

        for line in demographics_file:

            # remove the quoted names because they might contain commas
            line = re.sub('".*"', 'REMOVED', line)

            mrn, fullname, birthdate, gender = line.strip().split(',')
            mrn_holdout, holdout, cvgroup = holdout_file.readline().strip().split(',')
            assert(int(mrn) == int(mrn_holdout))
            birth_dts = datetime.fromisoformat(birthdate);

            # convert holdout to a boolean
            holdout = holdout.lower().strip() == 'true'

            # find the first matching line from the CAM file
            prior_date = datetime.fromordinal(1).date()
            while mrn != mrn_cam:
                mrn_cam, dts, measure, value = cam_file.readline().strip().split(',')
                assert(int(mrn) >= int(mrn_cam))

            # collect the CAM scores
            cam_scores = []
            ages = []
            cam_days = []
            while mrn == mrn_cam:
                if value.lower() == 'positive':
                    cam_scores.append(1)
                if value.lower() == 'negative':
                    cam_scores.append(0)

                ages.append((datetime.fromisoformat(dts) - birth_dts).days/365.25)

                # if this is a new date (starting at 5 am), add another cam day to the list
                date = (datetime.fromisoformat(dts) - timedelta(hours=5)).date()
                if date != prior_date:
                    cam_days.append(0)
                    prior_date = date

                # if this cam screen was positive, mark the day as positive
                if value.lower() == 'positive':
                    cam_days[-1] = 1

                # buffer the next line from the CAM file
                try:
                    mrn_cam, dts, measure, value = \
                        cam_file.readline().strip().split(',')
                except ValueError:
                    mrn_cam = -1 # end of file

            if len(cam_scores) > 0:
                all_tally.update(gender, cam_scores, ages, cam_days)
                if holdout:
                    holdout_tally.update(gender, cam_scores, ages, cam_days)
                else:
                    train_tally.update(gender, cam_scores, ages, cam_days)
                if sum(cam_scores) == 0:
                    cam_negative_tally.update(gender, cam_scores, ages, cam_days)
                else:
                    cam_positive_tally.update(gender, cam_scores, ages, cam_days)


    print(f'Measure\tAll Patients\tTraining Set\tTesting Set\tCAM positive\tCAM negative')
    rows = [
            ['Number of patients', lambda tally: f'{tally.count}'],
            ['Mean age (std dev)',
                lambda tally: f'{statistics.mean(tally.mean_ages):.1f} ' +
                    f'({statistics.stdev(tally.mean_ages):.1f})'],
            ['Male (%)', lambda tally: f'{tally.count_male} ' +
                f'({ 100 * tally.count_male / tally.count :.1f})'],
            ['Female (%)', lambda tally: f'{tally.count_female} ' +
                f'({ 100 * tally.count_female / tally.count :.1f})'],
            #['Median CAM evaluations per patient',
            #    lambda tally: f'{statistics.median(tally.cam_counts):0.1f}'],
            ['Patients with at least one positive CAM screen (%)',
                lambda tally:
                    f'{np.sum(np.array(tally.cam_positive_fraction)!=0)} ' +
                    f'({100*np.mean(np.array(tally.cam_positive_fraction)!=0):.1f})'
                    ],
            ['Mean CAM evaluations per patient (std dev)',
                lambda tally: f'{statistics.mean(tally.cam_counts):.1f} ' +
                    f'({statistics.stdev(tally.cam_counts):.1f})' ],
            ['Average percent positive CAM screens per patient (std dev)',
                lambda tally: f'{statistics.mean(tally.cam_positive_fraction) * 100:.1f} '
                    f'({statistics.stdev(tally.cam_positive_fraction) * 100:.1f})'],
            ['Mean CAM evaluation days per patient (std dev)',
                lambda tally: f'{statistics.mean(tally.cam_day_counts):.1f} ' +
                    f'({statistics.stdev(tally.cam_day_counts):.1f})' ],
            ['Average percent positive CAM days (std dev)',
                lambda tally: f'{statistics.mean(tally.cam_positive_day_fraction) * 100:.1f} '
                    f'({statistics.stdev(tally.cam_positive_day_fraction) * 100:.1f})'],
        ]
    for label, f in rows:
        print( '\t'.join([label]
            + [ f(tally) for tally in [all_tally, train_tally, holdout_tally,
                cam_positive_tally, cam_negative_tally] ]
            ))
