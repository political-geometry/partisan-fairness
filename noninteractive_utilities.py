
import csv
import pandas as pd
import numpy as np
import shapefile
import matplotlib.pyplot as plt
bar_plot_color = [0.36078431372549,0.682352941176471,0.611764705882353]
dt = 0.01
             

historical_elections = pd.read_csv('data/1976-2018-house.csv', encoding = "ISO-8859-1")  # Load MIT Election Lab Data
min_number_of_districts = 6

def get_eg_wasted_votes(republican_votes_by_district, democrat_votes_by_district):
    
    cumulative_wasted_r_votes = 0
    cumulative_wasted_d_votes = 0
    total_votes_in_state = np.sum(republican_votes_by_district) + np.sum(democrat_votes_by_district)
    
    for i in range(len(republican_votes_by_district)):
        district = i + 1
        votes_for_r = republican_votes_by_district[i]
        votes_for_d = democrat_votes_by_district[i]
        total_votes_in_district = republican_votes_by_district[i] + democrat_votes_by_district[i]
        needed_to_win = int(np.ceil(total_votes_in_district/2))
        
        if votes_for_r >= needed_to_win:
            wasted_by_r = votes_for_r - needed_to_win
            wasted_by_d = votes_for_d
        else:
            wasted_by_d = votes_for_d - needed_to_win
            wasted_by_r = votes_for_r
        
        cumulative_wasted_r_votes = cumulative_wasted_r_votes + wasted_by_r
        cumulative_wasted_d_votes = cumulative_wasted_d_votes + wasted_by_d
        
    return (cumulative_wasted_d_votes - cumulative_wasted_r_votes)/total_votes_in_state
    
def get_eg_seats_votes(republican_votes_by_district, democrat_votes_by_district):
    
    total_votes_in_state = np.sum(republican_votes_by_district) + np.sum(democrat_votes_by_district)
    vote_differential = np.array(republican_votes_by_district) - np.array(democrat_votes_by_district) 
    seats_won_by_republicans = len(np.where(vote_differential > 0)[0].flatten())
    total_republican_seat_share = seats_won_by_republicans/len(republican_votes_by_district)
    
    total_republican_vote_share = np.sum(republican_votes_by_district)/(total_votes_in_state)
    return total_republican_seat_share - 2*total_republican_vote_share + 0.5 

def make_table_comparing_eg_expressions(states,latex=False):
    year = 2016
    if not latex:
        row_format ="{:>15}" * (3)
        print(row_format.format("State", "EG", "S - 2V + 1/2"))
    else:
        row_format ="{} & " * (2)
        row_format = row_format + " {} \\\\ "
        print("\\begin{tabular}{c|rc} {\\bf State} & {\\bf EG} & $S-2V+\\frac 12$\\\\ \n \\hline")

    for state in states:
        rep_votes_by_district, dem_votes_by_district = get_two_party_votes(state, year)
        eg_wasted_votes = get_eg_wasted_votes(rep_votes_by_district, dem_votes_by_district)
        eg_seats_votes = get_eg_seats_votes(rep_votes_by_district, dem_votes_by_district)

        print(row_format.format(state,round(eg_wasted_votes,2),round(eg_seats_votes,2)))
    
    print("\\end{tabular}")


def ups_linear(vote_share_by_district):
    num_divisions = 1000
    vote_swings = np.arange(-1,1,1/num_divisions)

    votes = []
    seats = []
    mean_vote_share = np.mean(vote_share_by_district)
    
    for i in range(len(vote_swings)):
        
        delta_v = vote_swings[i]
        new_election = np.array(vote_share_by_district) + np.ones(np.shape(vote_share_by_district)) * delta_v
        
        # Handle unrealistic edge conditions outside the interval [0, 1]
        # If this is unsatisfying to you, check out ups_logit in utilities.py
        new_election[new_election < 0] = 0
        new_election[new_election > 1] = 1
        
        new_seats = len(np.where(np.array(new_election) > 0.5)[0]) / len(new_election)
        
        votes.append(np.mean(new_election))
        seats.append(new_seats)

    interpolated_votes = np.arange(0,1,1/num_divisions)
    interpolated_seats = np.floor(np.interp(interpolated_votes,votes, seats)*len(vote_share_by_district))/len(vote_share_by_district)

    return interpolated_votes, interpolated_seats


def get_two_party_votes(state, year):
    in_state = historical_elections['state_po'] == state
    in_year = historical_elections["year"] == year

    results_in_state_and_year = historical_elections[in_state & in_year]
    in_republican_party = results_in_state_and_year['party'] == "republican"
    in_democratic_party = (results_in_state_and_year['party'] == "democrat") | (results_in_state_and_year['party'] == "democratic-farmer-labor")
    not_a_write_in = results_in_state_and_year['writein'] == False
    
    number_of_districts = np.amax(results_in_state_and_year['district'])
    
    district_offset = 1
    if number_of_districts == 0:
        number_of_districts = 1
        district_offset = 0

    republican_votes_by_district = []
    democrat_votes_by_district = []

    for i in range(number_of_districts):
        in_district = results_in_state_and_year['district'] == i + district_offset

        republican_in_district = results_in_state_and_year[in_district & in_republican_party & not_a_write_in]
        republican_candidates = republican_in_district['candidate'].values

        if len(republican_candidates) == 0:
            print('Warning! No Republican candidate in ' + state + ' District ' + str(i + 1) + ' in ' + str(year))
            republican_votes_by_district.append(0)
        else:
            # The next lines are to handle the way New York's data is reported:
            matches_candidate_name = results_in_state_and_year['candidate'] == republican_in_district['candidate'].values[0]
            republican_in_district = results_in_state_and_year[in_district & matches_candidate_name & not_a_write_in]
            all_votes_for_candidate_str = republican_in_district['candidatevotes'].values
            all_votes_for_candidate = [int(vote) for vote in all_votes_for_candidate_str]
            republican_vote = np.sum(np.array(all_votes_for_candidate))
            republican_votes_by_district.append(republican_vote)

        democrat_in_district = results_in_state_and_year[in_district & in_democratic_party]
        democrat_candidates = democrat_in_district['candidate'].values

        if len(democrat_candidates) == 0:
            print('Warning! No Democrat candidate in ' + state + ' District ' + str(i + 1) + ' in ' + str(year))
            democrat_votes_by_district.append(0)
        else:
            # The next are to handle the way New York's data is reported:
            matches_candidate_name = results_in_state_and_year['candidate'] == democrat_in_district['candidate'].values[0]
            democrat_in_district = results_in_state_and_year[in_district & matches_candidate_name & not_a_write_in]
            all_votes_for_candidate_str = democrat_in_district['candidatevotes'].values
            all_votes_for_candidate = [int(vote) for vote in all_votes_for_candidate_str]
            democrat_vote = np.sum(np.array(all_votes_for_candidate))
            democrat_votes_by_district.append(democrat_vote)

        if republican_votes_by_district[-1] == 0 and democrat_votes_by_district[-1] == 0:
            print('Warning! No votes registered in ' + state + ' District ' + str(i + 1) + ' in ' + str(year))

    return republican_votes_by_district, democrat_votes_by_district

    
# Get the fraction of votes each party got, when the race is reduced to just two parties. 
def votes_to_shares_by_district(party_a_votes, party_b_votes):
    
    party_a_vote_share = []
    party_b_vote_share = []
    
    for i in range(len(party_a_votes)):
        total_votes = party_a_votes[i] + party_b_votes[i]
        if total_votes != 0:
            party_a_vote_share.append(party_a_votes[i]/(total_votes))
            party_b_vote_share.append(party_b_votes[i]/(total_votes))
        else:
            print('Warning! No votes registered in a district. District was omitted.')

    return party_a_vote_share, party_b_vote_share


# Get the vote and seat shares for party A
def district_vote_shares_to_vote_and_seat_shares(vote_shares_by_district):
    
    # Note! This is the average district vote share, not the vote share in the state
    # They would agree if turnout everywhere were equal.
    vote_share = np.mean(vote_shares_by_district)  
    seats_won = 0
    seat_total = 0
    
    for vote_share_in_district in vote_shares_by_district:
        seat_total = seat_total + 1
        if vote_share_in_district > 0.5:
            seats_won = seats_won + 1
    
    seat_share = seats_won/seat_total
    return vote_share, seat_share



import pandas as pd

def read_daily_kos_data(include_loser_bonus_states=True):
    year_to_row = {2016: 3, 2012: 5, 2008: 7}
    result_row = 2
        
    votes = []
    seats = []
    
    loser_bonus_state_count = 0

    for year in [2012, 2016]: #[2008, 2012, 2016]:
        presidential_dictionary = {}
        state_election_dictionary = {}

        with open('data/daily_kos.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count < 2:
                    pass
                else:
                    state_abbreviation = row[0]
                    split_string = state_abbreviation.split('-')
                    two_letter_state = split_string[0]

                    presidential_results = [float(row[year_to_row[year]]), float(row[year_to_row[year] + 1])]

                    result = row[result_row]
                    result_code = -1
                    if result == '(R)':
                        result_code = 0
                    if result == '(D)':
                        result_code = 1

                    if two_letter_state in state_election_dictionary:
                        state_election_dictionary[two_letter_state].append(result_code)
                    else:
                        state_election_dictionary[two_letter_state] = [result_code]

                    if two_letter_state in presidential_dictionary:
                        presidential_dictionary[two_letter_state].append(presidential_results)
                    else:
                        presidential_dictionary[two_letter_state] = [presidential_results]

                line_count += 1

        for key in presidential_dictionary.keys():
            all_presidential_results = presidential_dictionary[key]
            all_state_results = state_election_dictionary[key]

            republican_vote_share = 0
            for district in all_presidential_results:
                republican_vote_share = republican_vote_share + district[1]

            republican_vote_share = republican_vote_share / len(all_presidential_results)
            republican_vote_share = republican_vote_share / 100.0

            republican_seat_share = 0
            for district in all_state_results:
                if district == 0:
                    republican_seat_share = republican_seat_share + 1

            republican_seat_share = republican_seat_share / len(all_state_results)

            if len(all_state_results) >= min_number_of_districts:
                if (republican_vote_share < 0.5 and republican_seat_share < 0.5) or (1 - republican_vote_share < 0.5 and 1 - republican_seat_share < 0.5):
                    votes.append(republican_vote_share)
                    seats.append(republican_seat_share)
                    
                else:
                    if include_loser_bonus_states:
                        votes.append(republican_vote_share)
                        seats.append(republican_seat_share)
                        loser_bonus_state_count = loser_bonus_state_count + 1
    return votes, seats



def get_precinct_population_and_voting_data(state):
        
    if state == "MD":
        data = shapefile.Reader('data/MD-shapefiles/MD_precincts_abs.shp')  # From: https://github.com/mggg-states/GA-shapefiles
        pop_index = 9
        republican_index = 28
        democrat_index = 29

    if state == "GA":
        data = shapefile.Reader('data/GA-shapefiles/GA_precincts16.shp')  # From https://github.com/mggg-states/GA-shapefiles
        pop_index = 13
        republican_index = 8
        democrat_index = 7

    if state == "TX":
        data = shapefile.Reader('data/TX_vtds/TX_vtds.shp')  # From https://github.com/mggg-states/TX-shapefiles
        pop_index = 5
        republican_index = 16
        democrat_index = 17

    populations_and_republican_vote_shares = []
    for record in data.shapeRecords():
        dem_and_rep_votes = int(record.record[republican_index]) + int(record.record[democrat_index])
        if dem_and_rep_votes > 0:
            precinct = [int(record.record[pop_index]), int(record.record[republican_index])/dem_and_rep_votes]
            populations_and_republican_vote_shares.append(precinct)

    populations_and_republican_vote_shares = np.array(populations_and_republican_vote_shares)
    return populations_and_republican_vote_shares


def get_histogram(state):
    populations_and_republican_vote_shares = get_precinct_population_and_voting_data(state)
    histogram_bins = np.arange(0,1 + dt,dt)

    total_population = []

    for i in range(0,len(histogram_bins) - 1):
        lower_bound = histogram_bins[i]
        upper_bound = histogram_bins[i + 1]
        indices_in_range = np.argwhere((populations_and_republican_vote_shares[:,1] >= lower_bound) & (populations_and_republican_vote_shares[:,1] < upper_bound)).flatten()
        total_population.append(np.sum(populations_and_republican_vote_shares[indices_in_range,0]))


    total_population = np.array(total_population)
    histogram_centers = np.arange(0 + dt/2, 1 + dt/2, dt)
    return histogram_centers, total_population

def get_population_in_range(lower_bound, upper_bound, histogram_centers, total_population):
    population_in_range = 0
    
    for i in range(len(histogram_centers)):
        vote_share = histogram_centers[i]
        if vote_share >= lower_bound and vote_share < upper_bound:
            population_in_range = population_in_range + total_population[i]
    return population_in_range

def get_population_in_range_for_state(state,lower_bound,upper_bound):
    histogram_centers, total_population = get_histogram(state)
    population_in_range = get_population_in_range(lower_bound, upper_bound, histogram_centers, total_population)
    return population_in_range
    
def plot_precinct_population_histogram(state):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    histogram_centers, total_population = get_histogram(state)
    plt.bar(histogram_centers,total_population, dt,color=bar_plot_color)
    
    ax.text(0,1.05*np.amax(total_population),state,fontsize=big_text_size,fontname=font_name)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xlabel("Republican vote share in precincts", fontname=font_name, fontsize=font_size)
    ax.set_ylabel("Total population in precincts", fontname=font_name, fontsize=font_size)

def plot_histogram_row(states):
    grid_color = [0.5, 0.5, 0.5, 0.5]
    font_name = 'Helvetica'

    N = len(states)
    scalar = 1.05
    fig, axs = plt.subplots(1, N, sharex=True,figsize=(10,2.5),dpi=500)
    fig.tight_layout()
    for i in range(N):
        ax = axs[i]
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        histogram_centers, total_population = get_histogram(states[i])

        ax.bar(histogram_centers,total_population, dt,color=bar_plot_color)
        
        ax.plot([0.15, 0.15],[0, scalar*np.amax(total_population)], '--', color=grid_color)
        ax.plot([0.85, 0.85],[0, scalar*np.amax(total_population)], '--', color=grid_color)

        ax.text(0,scalar*np.amax(total_population),states[i],fontsize=15,fontname=font_name)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_xlabel("Republican vote share in precincts", fontname=font_name, fontsize=8)
        
        if i == 0:
            ax.set_ylabel("Total population in precincts", fontname=font_name, fontsize=8)
            




