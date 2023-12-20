import numpy as np
from scipy.signal import find_peaks, medfilt
import matplotlib.pyplot as plt

class Ventilatory:
    @staticmethod
    def get_cycles(vpt_data):
        distance_peaks = lambda pt1, pt2: abs(pt1 - pt2)
        should_discard_terminal_peak = lambda volume_curve, pt1, pt2: \
            abs(volume_curve[pt2]*0.95) < abs(volume_curve[pt1]) < abs(volume_curve[pt2]*1.05) and \
            distance_peaks(pt1, pt2) < 50

        def find_respiratory_peaks(volume_curve, show_results=False):
            max_value = np.max(volume_curve)
            inv_volume_curve = max_value-volume_curve

            inspiratory_peaks, _ = find_peaks(volume_curve, prominence=200)
            expiratory_peaks, _ = find_peaks(inv_volume_curve, prominence=200)

            mean_peak_value = np.mean(volume_curve[inspiratory_peaks])
            first_cycle_pt = np.argmin(volume_curve[:inspiratory_peaks[0]])
            last_cycle_pt = np.argmin(volume_curve[inspiratory_peaks[-1]:]) + inspiratory_peaks[-1]

            if (should_discard_terminal_peak(volume_curve, first_cycle_pt, expiratory_peaks[0])):
                first_peak = np.array([], dtype=np.uint8)
            else:
                first_peak = np.array([first_cycle_pt])

            if (should_discard_terminal_peak(volume_curve, last_cycle_pt, expiratory_peaks[-1])):
                last_peak = np.array([], dtype=np.uint8)
            else:
                last_peak = np.array([last_cycle_pt])

            expiratory_peaks = np.concatenate((first_peak, expiratory_peaks, last_peak))
            
            if show_results:
                plt.plot(volume_curve, color='blue', linestyle='solid')
                plt.plot(expiratory_peaks, volume_curve[expiratory_peaks], "xr")
                plt.hlines(mean_peak_value, 0, len(volume_curve))
                plt.show()

            return inspiratory_peaks, expiratory_peaks
        
        vpt_separated_data = vpt_data.get_data(separated_vars=True)
        volume_curve_smoothed = medfilt(vpt_separated_data["volume"], kernel_size=101)

        inspiratory_peaks, expiratory_peaks = find_respiratory_peaks(volume_curve_smoothed, show_results=False)

        # flow_curve = medfilt(vpt_separated_data["flow"], kernel_size=21)
        # plt.plot(flow_curve, color='blue', linestyle='solid')
        # plt.plot(volume_curve_smoothed, color='green', linestyle='dashed')
        # plt.plot(expiratory_peaks, flow_curve[expiratory_peaks], "xr")
        # plt.plot(inspiratory_peaks, flow_curve[inspiratory_peaks], "xc")
        # plt.show()

        return expiratory_peaks