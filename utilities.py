import matplotlib as mpl
import matplotlib.pyplot as plt

import math
import numpy as np
from scipy import stats

dpi = 100
mpl.rcParams['figure.dpi']= dpi

background_line_width = 1.0
ups_line_width = 2.0
ups_color = [0.8, 0.2, 0.9]
ups_color_reversed = [0.8, 0.2, 0.9, 0.2]
ups_color_shaded = [0.9, 0.9, 0.9, 0.8]
grid_color = [0.5, 0.5, 0.5, 0.5]
regression_color = [0.898039215686275,0.376470588235294,0.294117647058824]
actual_color = [0.2, 0.6, 0.2, 1.0]
bias_color = [0.92156862745098,0.733333333333333,0.168627450980392]
mean_median_color = [0.352941176470588,0.694117647058824,0.803921568627451]
partisan_symmetry_color = [0.898039215686275,0.376470588235294,0.294117647058824]
eg_color = [0.95, 0.65, 0.15, 1.0]
eg_color_transparent = [1.0, 0.7, 0.2, 0.6]

x_buff = 0.05
y_buff = 0.05
metric_buffer = 0.008

x_range = [-x_buff, 1 + x_buff]
y_range = [-y_buff, 1 + y_buff]
font_name = 'Helvetica'
font_size = 15
big_text_size = 32


def compute_mean_median_intercept(votes, seats):
    v_intersect = np.interp(0.5, seats, votes)
    indices_where_votes_equal_point_five = np.argwhere(votes == 0.5)
    if len(indices_where_votes_equal_point_five) > 0:
        if seats[indices_where_votes_equal_point_five[0][0]] == 0.5:
            v_intersect = 0.5
    return v_intersect

def compute_partisan_bias_intercept(votes, seats):
    s_intersect = np.interp(0.5, votes, seats)
    return s_intersect

def compute_symmetric_intercept(actual_vote_share, actual_seat_share, votes, seats):
    symmetric_v = 1 - actual_vote_share
    symmetric_s = 1 - actual_seat_share
    s_intersect_symmetric_interp = np.interp(symmetric_v, votes, seats)

    if symmetric_s - s_intersect_symmetric_interp < 0:
        s_intersect_symmetric = np.where(seats - s_intersect_symmetric_interp >= 0, seats, np.inf).min()

    if symmetric_s - s_intersect_symmetric_interp > 0:
        s_intersect_symmetric = np.where(seats - s_intersect_symmetric_interp <= 0, seats, -np.inf).max()

    if symmetric_s == s_intersect_symmetric_interp:
        s_intersect_symmetric = s_intersect_symmetric_interp
    return symmetric_s, s_intersect_symmetric

def compute_signed_area_between_curves(votes, seats):
    reversed_votes = np.concatenate((np.flip(1 - np.array(votes[:-1])), np.array([1])))
    reversed_seats = np.flip(1 - np.array(seats[:]))
    cumulative_sum = 0
    for i in range(len(reversed_votes) - 2):
        if 0.4 <= votes[i + 1] <= 0.6:
            area = np.abs(reversed_votes[i] - votes[i+1])*np.abs(reversed_seats[i] - seats[i+2])

            if reversed_seats[i] <= seats[i+2]:
                cumulative_sum = cumulative_sum + area
            else:
                cumulative_sum = cumulative_sum - area
    return cumulative_sum

def compute_unsigned_area_between_curves(votes, seats):
    reversed_votes = np.concatenate((np.flip(1 - np.array(votes[:-1])), np.array([1])))
    reversed_seats = np.flip(1 - np.array(seats[:]))
    cumulative_sum = 0
    for i in range(len(reversed_votes) - 2):
        area = np.abs(reversed_votes[i] - votes[i+1])*np.abs(reversed_seats[i] - seats[i+2])
        cumulative_sum = cumulative_sum + area
    return cumulative_sum



def add_ups_curve(ax, votes, seats):
    ax.step(votes, seats, color=ups_color, linewidth=ups_line_width, zorder=1)

def add_actual_election(ax, actual_vote_share, actual_seat_share, point_label=""):
    ax.scatter([actual_vote_share], [actual_seat_share], color=actual_color,edgecolors=actual_color, zorder=3, s=50)
    ax.annotate(point_label,(actual_vote_share,actual_seat_share), textcoords="offset points", xytext=(0,10), ha='center',fontname=font_name)

def add_mean_median(ax, votes, seats):
    v_intersect = compute_mean_median_intercept(votes, seats)
    mean_median_x = [v_intersect, v_intersect, 0.5, 0.5]
    mean_median_y = [0.5 - metric_buffer, 0.5 + metric_buffer, 0.5 + metric_buffer, 0.5 - metric_buffer]

    ax.fill(mean_median_x, mean_median_y, '--', color=mean_median_color, edgecolor=mean_median_color,
                     zorder=2)

def add_partisan_bias(ax, votes, seats):
    s_intersect = compute_partisan_bias_intercept(votes, seats)
    bias_x = [0.5 - metric_buffer, 0.5 + metric_buffer, 0.5 + metric_buffer,
                  0.5 - metric_buffer]
    bias_y = [s_intersect, s_intersect, 0.5, 0.5]
        
    ax.fill(bias_x, bias_y, '--', color=bias_color, edgecolor=bias_color, zorder=2)


def add_partisan_symmetry(ax,actual_vote_share, actual_seat_share, votes, seats):
    symmetric_s, s_intersect_symmetric = compute_symmetric_intercept(actual_vote_share, actual_seat_share, votes, seats)
    partisan_symmetry_x = [actual_vote_share - metric_buffer, actual_vote_share + metric_buffer, actual_vote_share + metric_buffer,
                           actual_vote_share - metric_buffer]
    partisan_symmetry_y = [1-s_intersect_symmetric, 1-s_intersect_symmetric, 1-symmetric_s, 1-symmetric_s]
    ax.fill(partisan_symmetry_x, partisan_symmetry_y, '--', color=partisan_symmetry_color,edgecolor=partisan_symmetry_color, zorder=2)
    
    
def add_flipped_and_shaded_curve(ax, votes, seats):
    reversed_votes = np.concatenate((np.flip(1 - np.array(votes[:-1])), np.array([1])))
    reversed_seats = np.flip(1 - np.array(seats[:]))

    ax.step(reversed_votes, reversed_seats, color=ups_color_reversed, linewidth=ups_line_width, zorder=1)
    for i in range(len(reversed_votes) - 2):
        x = [reversed_votes[i], reversed_votes[i], votes[i+1], votes[i+1]]
        y = [reversed_seats[i], seats[i+2], seats[i+2], reversed_seats[i]]
        ax.fill(x, y, facecolor=ups_color_shaded)

def add_regression_line(ax, votes, seats):
    slope, intercept, r_value, p_value, std_err = stats.linregress(votes, seats)
    print("Slope of regression line: " + str(slope))
    print("Intercept of regression line: " + str(intercept))

    ax.plot(np.array(x_range), intercept + np.array(x_range) * slope, ':', color=regression_color, zorder=4)

def plot_mean_mean_and_partisan_bias(actual_vote_share, actual_seat_share, votes, seats, text=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    add_actual_election(ax, actual_vote_share, actual_seat_share)
    add_ups_curve(ax, votes, seats)
    add_mean_median(ax, votes, seats)
    add_partisan_bias(ax, votes, seats)
    configure_plot(plt,ax,text)

def plot_ups(actual_vote_share, actual_seat_share, votes, seats, text=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    add_actual_election(ax, actual_vote_share, actual_seat_share)
    add_ups_curve(ax, votes, seats)
    configure_plot(plt,ax,text)

def plot_vote_and_seat_shares(vote_share,seat_share, text=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    add_actual_election(ax, vote_share, seat_share)
    configure_plot(plt,ax,text)

def plot_list_of_vote_and_seat_shares(vote_share_list,seat_share_list, regression=True, text='',point_labels=[]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if len(point_labels) == 0:  # If we don't have any labels for the points
        point_labels = " " * len(vote_share_list)
    
    for i in range(0,len(vote_share_list)):
        add_actual_election(ax, vote_share_list[i], seat_share_list[i], point_labels[i])
    
    if regression:
        add_regression_line(ax, vote_share_list, seat_share_list)
    configure_plot(plt,ax,text)
    
def plot_ups_grid(labels,actual_vote_share_list, actual_seat_share_list, vote_list, seat_list):
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    for i in range(0,len(labels)):
        ax = axs[math.floor(i/2),i % 2]
        rep_vote_share = actual_vote_share_list[i]
        rep_seat_share = actual_seat_share_list[i]
        votes = vote_list[i]
        seats = seat_list[i]
        add_actual_election(ax, rep_vote_share, rep_seat_share)
        add_ups_curve(ax, votes, seats)
        configure_plot(plt,axs[math.floor(i/2),i % 2], labels[i])

def plot_mean_median_and_partisan_bias_grid(labels,actual_vote_share_list, actual_seat_share_list, vote_list, seat_list):
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    for i in range(0,len(labels)):
        ax = axs[math.floor(i/2),i % 2]
        rep_vote_share = actual_vote_share_list[i]
        rep_seat_share = actual_seat_share_list[i]
        votes = vote_list[i]
        seats = seat_list[i]
        add_actual_election(ax, rep_vote_share, rep_seat_share)
        add_ups_curve(ax, votes, seats)
        add_mean_median(ax, votes, seats)
        add_partisan_bias(ax, votes, seats)
        configure_plot(plt,axs[math.floor(i/2),i % 2], labels[i])
        
def plot_symmetric_point_and_shading_grid(labels,actual_vote_share_list, actual_seat_share_list, vote_list, seat_list):
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    for i in range(0,len(labels)):
        ax = axs[math.floor(i/2),i % 2]
        rep_vote_share = actual_vote_share_list[i]
        rep_seat_share = actual_seat_share_list[i]
        votes = vote_list[i]
        seats = seat_list[i]
        add_actual_election(ax, rep_vote_share, rep_seat_share)
        add_ups_curve(ax, votes, seats)
        add_partisan_symmetry(ax, rep_vote_share, rep_seat_share, votes, seats)
        add_flipped_and_shaded_curve(ax, votes, seats)
        configure_plot(plt,axs[math.floor(i/2),i % 2], labels[i])

def plot_all_measures_grid(labels,actual_vote_share_list, actual_seat_share_list, vote_list, seat_list):
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    for i in range(0,len(labels)):
        ax = axs[math.floor(i/2),i % 2]
        rep_vote_share = actual_vote_share_list[i]
        rep_seat_share = actual_seat_share_list[i]
        votes = vote_list[i]
        seats = seat_list[i]
        add_actual_election(ax, rep_vote_share, rep_seat_share)
        add_ups_curve(ax, votes, seats)
        add_partisan_symmetry(ax, rep_vote_share, rep_seat_share, votes, seats)
        add_mean_median(ax, votes, seats)
        add_partisan_bias(ax, votes, seats)
        add_flipped_and_shaded_curve(ax, votes, seats)
        configure_plot(plt,axs[math.floor(i/2),i % 2], labels[i])


def plot_eg_band():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    band_width = 0.04
    max_x_lower = (0 - 0.5) * (0.5) + band_width + 0.5
    min_x_lower = (0 - 0.5) * (0.5) - band_width + 0.5
    max_x_upper = (1 - 0.5) * (0.5) + band_width + 0.5
    min_x_upper = (1 - 0.5) * (0.5) - band_width + 0.5

    efficiency_band_x = [min_x_lower, max_x_lower, max_x_upper, min_x_upper]
    efficiency_band_y = [0, 0, 1, 1]

    plt.fill(efficiency_band_x, efficiency_band_y, '--', color=eg_color_transparent, edgecolor=eg_color_transparent, zorder=2)
    v_actual = 0.65
    s_actual = 0.65
    plt.text(v_actual + 0.05, s_actual, '(0.65, 0.65)', fontsize=12, fontname=font_name)
    plt.text(v_actual + 0.06, s_actual - .08, 'EG = -0.15', fontsize=12, fontname=font_name)
    plt.text(0.05, 0.9, '|EG| < 0.08', fontsize=16, color=eg_color, fontname=font_name)
    ax.set_aspect('equal', 'box')

    add_actual_election(ax, v_actual, s_actual)
    configure_plot(plt,ax)
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))



def configure_plot(plt,ax,text=''):
    ax.plot(x_range, [0.5, 0.5], color=grid_color, linewidth=background_line_width, zorder=-1)
    ax.plot([0.5, 0.5], y_range, color=grid_color, linewidth=background_line_width, zorder=-1)
    ax.plot(x_range, y_range, '--', color=grid_color, linewidth=background_line_width, zorder=0)

    ax.text(0.05,0.9,text,fontsize=big_text_size,fontname=font_name)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel("Average district Republican vote share", fontname=font_name, fontsize=font_size)
    ax.set_ylabel("Republican seat share", fontname=font_name, fontsize=font_size)

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1.0], fontsize=13, fontname=font_name)
    ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1.0], fontsize=13, fontname=font_name)
    ax.set_xlim((-x_buff,1+x_buff))
    ax.set_ylim((-y_buff,1+y_buff))


def make_partisan_symmetry_table(labels,actual_vote_share_list, actual_seat_share_list, vote_list, seat_list, latex=False):
    N = 5
    if not latex:
        row_format ="{:>15}" * (N)
        print(row_format.format("Election", "beta-average", "beta(Vo)", "beta(0.5)", "Mean-median"))
    else:
        row_format ="{} & " * (N-1)
        row_format = row_format + " {} \\\\ "
        print("\\begin{tabular}{c|c|c|c|c}\\n State & $\\beta$-average & $\\beta(\\VO)$ & $\\beta(.5)$ & Mean-median \\\\")
        print("& on $[0.4,0.6]$  &              &            & score\\\\")
    
    for i in range(0,len(labels)):
        rep_vote_share = actual_vote_share_list[i]
        rep_seat_share = actual_seat_share_list[i]
        votes = vote_list[i]
        seats = seat_list[i]

        mean_median = 0.5 - compute_mean_median_intercept(votes, seats)
        partisan_bias = 0.5 - compute_partisan_bias_intercept(votes, seats)
        
        symmetric_s, s_intersect_symmetric = compute_symmetric_intercept(rep_vote_share, rep_seat_share, votes, seats)
        partisan_symmetry = symmetric_s - s_intersect_symmetric
        signed_area = compute_signed_area_between_curves(votes, seats)
        unsigned_area = compute_unsigned_area_between_curves(votes,seats)
        print(row_format.format(labels[i], str(round(unsigned_area,2)), str(round(partisan_symmetry,2)), str(round(partisan_bias,2)), str(round(mean_median,2))))
        
    if latex:
        print("\\hline \n \\end{tabular}")

def make_competitiveness_table_for_state(republican_votes, democrat_votes, latex=False):
    N = 6
    
    if not latex:
        row_format ="{:>15}" * (N)
        print(row_format.format("District", "Total votes", "% Republican", "Winner", "Margin in %", "in votes"))
    else:
        row_format ="{} & {} & {}\\% & {} & {}\\% & {} \\\\"
        print("\\begin{tabular}{c|c|c|c|c|c} \n &  &   &   &   \\multicolumn{2}{c}{\\textbf{Margin of victory...}} \\\\")
        print("\\textbf{District}   &  \\textbf{Total votes}   &   \\textbf{\\% Republican} &  \\textbf{Winner} & \\bf{in \\%}  &   \\textbf{in votes}\\\\ \n \\hline")

    cumulative_votes = 0
    cumulative_republican_votes = 0
    d_districts = 0
    r_districts = 0
    
    for i in range(len(republican_votes)):
        district = i + 1
        total_votes = republican_votes[i] + democrat_votes[i]
        percent_republican = round(100*republican_votes[i] /total_votes,1)
        margin_of_victory_votes = round(np.abs(republican_votes[i] - democrat_votes[i]),2)
        margin_of_victory_percent = round(100*margin_of_victory_votes/total_votes,1)
        
        cumulative_votes = cumulative_votes + total_votes
        cumulative_republican_votes = cumulative_republican_votes +  republican_votes[i]

        total_votes = '{:,}'.format(total_votes)
        margin_of_victory_votes = '{:,}'.format(margin_of_victory_votes)

        if republican_votes[i] > democrat_votes[i]:
            winner = "R"
            r_districts = r_districts + 1

        else:
            winner = "D"
            d_districts = d_districts + 1
        
        print(row_format.format(district, total_votes, percent_republican, winner, margin_of_victory_percent, margin_of_victory_votes))
    
    if latex:
        print("\\hline \n Total & {} & {}\\% & {}R/{}D &   &  \\\\".format(round(cumulative_votes,2), round(100*cumulative_republican_votes/cumulative_votes,2), r_districts, d_districts))
        print("\\end{tabular}")


def make_wasted_votes_table_for_state(republican_votes, democrat_votes, latex=False):
    N = 7
    
    if not latex:
        row_format ="{:>15}" * (N)
        print(row_format.format("District", "Votes for R", "Votes for D", "Total R + D", "Needed to win", "Wasted by R", "Wasted by D"))
    else:
        row_format ="{} & " * (N-1)
        row_format = row_format + " {} \\\\ "
        print("\\begin{tabular}{r|rrrrrr} {\\bf District} & {\\bf Votes for R} & {\\bf Votes for D} & {\\bf Total votes} & {\\bf Needed to win}& {\\bf Wasted by R} & {\\bf Wasted by D} \\\\ \\hline")
    
    cumulative_wasted_r_votes = 0
    cumulative_wasted_d_votes = 0

    for i in range(len(republican_votes)):
        district = i + 1
        votes_for_r = republican_votes[i]
        votes_for_d = democrat_votes[i]
        total_votes = republican_votes[i] + democrat_votes[i]
        needed_to_win = int(np.ceil(total_votes/2))
        
        if votes_for_r >= needed_to_win:
            wasted_by_r = votes_for_r - needed_to_win
            wasted_by_d = votes_for_d
        else:
            wasted_by_d = votes_for_d - needed_to_win
            wasted_by_r = votes_for_r
        
        cumulative_wasted_r_votes = cumulative_wasted_r_votes + wasted_by_r
        cumulative_wasted_d_votes = cumulative_wasted_d_votes + wasted_by_d

        total_votes = '{:,}'.format(total_votes)
        votes_for_r = '{:,}'.format(votes_for_r)
        votes_for_d = '{:,}'.format(votes_for_d)
        needed_to_win = '{:,}'.format(needed_to_win)
        wasted_by_r = '{:,}'.format(wasted_by_r)
        wasted_by_d = '{:,}'.format(wasted_by_d)

        print(row_format.format(district,  votes_for_r, votes_for_d, total_votes, needed_to_win, wasted_by_r, wasted_by_d))
        
    total_election_votes = int(np.sum(republican_votes))+int(np.sum(democrat_votes))
    
    if not latex:
        print(row_format.format("Total",  '{:,}'.format(int(np.sum(republican_votes))), '{:,}'.format(int(np.sum(democrat_votes))), '{:,}'.format(total_election_votes), "", '{:,}'.format(cumulative_wasted_r_votes), '{:,}'.format(cumulative_wasted_d_votes)))
    else:
        print("\\hline")
        print("\\textit{{Total}} & {} & {} & {} & & {} & {}".format('{:,}'.format(int(np.sum(republican_votes))), '{:,}'.format(int(np.sum(democrat_votes))), '{:,}'.format(total_election_votes), '{:,}'.format(cumulative_wasted_r_votes), '{:,}'.format(cumulative_wasted_d_votes)))
        print("\\end{tabular}")

    print('\n Efficiency gap: ',round((cumulative_wasted_d_votes - cumulative_wasted_r_votes)/total_election_votes,4))



def save_plot(save_name_str):
    plt.gca()
    plt.savefig("outputs/" + save_name_str + '.png', dpi=500)
    import tikzplotlib
    tikzplotlib.clean_figure()
    tikzplotlib.get_tikz_code()

#    plt.savefig("outputs/" + save_name_str + '.pgf')
