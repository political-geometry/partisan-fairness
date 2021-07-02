import csv
import pandas as pd
import numpy as np

# This file contains reproductions of functions that are defined within Notebooks.
# It is used in make_figures_for_chapter.nb to collect and run all relevant scripts needed for figures

bar_plot_color = [0.36078431372549, 0.682352941176471, 0.611764705882353]
dt = 0.01

historical_elections = pd.read_csv('data/1976-2018-house.csv', encoding="ISO-8859-1")  # Load MIT Election Lab Data
min_number_of_districts = 6


def get_eg_wasted_votes(republican_votes_by_district, democrat_votes_by_district):
    cumulative_wasted_r_votes = 0
    cumulative_wasted_d_votes = 0
    total_votes_in_state = np.sum(republican_votes_by_district) + np.sum(democrat_votes_by_district)

    for i in range(len(republican_votes_by_district)):
        votes_for_r = republican_votes_by_district[i]
        votes_for_d = democrat_votes_by_district[i]
        total_votes_in_district = republican_votes_by_district[i] + democrat_votes_by_district[i]
        needed_to_win = int(np.ceil(total_votes_in_district / 2))

        if votes_for_r >= needed_to_win:
            wasted_by_r = votes_for_r - needed_to_win
            wasted_by_d = votes_for_d
        else:
            wasted_by_d = votes_for_d - needed_to_win
            wasted_by_r = votes_for_r

        cumulative_wasted_r_votes = cumulative_wasted_r_votes + wasted_by_r
        cumulative_wasted_d_votes = cumulative_wasted_d_votes + wasted_by_d

    return (cumulative_wasted_d_votes - cumulative_wasted_r_votes) / total_votes_in_state


def get_eg_seats_votes(republican_votes_by_district, democrat_votes_by_district):
    total_votes_in_state = np.sum(republican_votes_by_district) + np.sum(democrat_votes_by_district)
    vote_differential = np.array(republican_votes_by_district) - np.array(democrat_votes_by_district)
    seats_won_by_republicans = len(np.where(vote_differential > 0)[0].flatten())
    total_republican_seat_share = seats_won_by_republicans / len(republican_votes_by_district)

    total_republican_vote_share = np.sum(republican_votes_by_district) / (total_votes_in_state)
    return total_republican_seat_share - 2 * total_republican_vote_share + 0.5


def make_table_comparing_eg_expressions(states, latex=False):
    year = 2016
    if not latex:
        row_format = "{:>15}" * (3)
        print(row_format.format("State", "EG", "S - 2V + 1/2"))
    else:
        row_format = "{} & " * (2)
        row_format = row_format + " {} \\\\ "
        print("\\begin{tabular}{c|rc} {\\bf State} & {\\bf EG} & $S-2V+\\frac 12$\\\\ \n \\hline")

    for state in states:
        rep_votes_by_district, dem_votes_by_district = get_two_party_votes(state, year)
        eg_wasted_votes = get_eg_wasted_votes(rep_votes_by_district, dem_votes_by_district)
        eg_seats_votes = get_eg_seats_votes(rep_votes_by_district, dem_votes_by_district)

        print(row_format.format(state, round(eg_wasted_votes, 2), round(eg_seats_votes, 2)))

    print("\\end{tabular}")


def ups_linear(vote_share_by_district):
    num_divisions = 1000
    dt = 1/num_divisions

    vote_swings = np.linspace(-1, 1 + dt, num_divisions)

    votes = []
    seats = []

    for i in range(len(vote_swings)):
        delta_v = vote_swings[i]
        new_election = np.array(vote_share_by_district) + np.ones(np.shape(vote_share_by_district)) * delta_v

        # Handle unrealistic edge conditions outside the interval [0, 1]
        # If this is unsatisfying to you, check out ups_logit in utilities.py
        new_election[new_election < 0] = 0
        new_election[new_election > 1] = 1

        new_seats = len(np.where(np.array(new_election) > 0.5)[0])

        votes.append(np.mean(new_election))
        seats.append(new_seats)

    interpolated_votes = np.linspace(0, 1 + dt, num_divisions)
    interpolated_seats = np.floor(np.interp(interpolated_votes, votes, seats)) / len(vote_share_by_district)

    return interpolated_votes, interpolated_seats


def ups_linear_using_statewide_vote_share(republican_votes_by_district, democrat_votes_by_district):
    num_divisions = 1000
    dt = 1/num_divisions
    vote_swings = np.linspace(-1, 1 + dt, num_divisions)

    votes = []
    seats = []

    vote_share_by_district, *_ = votes_to_shares_by_district(republican_votes_by_district, democrat_votes_by_district)
    republican_statewide_vote_share, *_ = votes_to_overall_vote_share(republican_votes_by_district,
                                                                      democrat_votes_by_district)

    for i in range(len(vote_swings)):
        delta_v = vote_swings[i]
        new_election = np.array(vote_share_by_district) + np.ones(np.shape(vote_share_by_district)) * delta_v

        # Handle unrealistic edge conditions outside the interval [0, 1]
        # If this is unsatisfying to you, check out ups_logit in utilities.py
        new_election[new_election < 0] = 0
        new_election[new_election > 1] = 1

        new_seats = len(np.where(np.array(new_election) > 0.5)[0]) / len(new_election)

        votes.append(republican_statewide_vote_share + delta_v)
        seats.append(new_seats)

    interpolated_votes = np.linspace(0, 1 + dt, num_divisions)
    interpolated_seats = np.floor(np.interp(interpolated_votes, votes, seats) * len(vote_share_by_district)) / len(
        vote_share_by_district)

    return interpolated_votes, interpolated_seats


def get_two_party_votes(state, year):
    in_state = historical_elections['state_po'] == state
    in_year = historical_elections["year"] == year

    results_in_state_and_year = historical_elections[in_state & in_year]
    in_republican_party = results_in_state_and_year['party'] == "republican"
    in_democratic_party = (results_in_state_and_year['party'] == "democrat") | (
            results_in_state_and_year['party'] == "democratic-farmer-labor")
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
            # The next lines are to handle the way New York's data is reported
            # In New York, the same candidate can run for multiple parties
            matches_candidate_name = results_in_state_and_year['candidate'] == \
                                     republican_in_district['candidate'].values[0]
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
            # The next are to handle the way New York's data is reported
            # In New York, the same candidate can run for multiple parties
            matches_candidate_name = results_in_state_and_year['candidate'] == democrat_in_district['candidate'].values[
                0]
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
            party_a_vote_share.append(party_a_votes[i] / (total_votes))
            party_b_vote_share.append(party_b_votes[i] / (total_votes))
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

    seat_share = seats_won / seat_total
    return vote_share, seat_share


def read_daily_kos_data(include_loser_bonus_states=True):
    year_to_row = {2016: 3, 2012: 5, 2008: 7}
    result_row = 2

    votes = []
    seats = []

    loser_bonus_state_count = 0

    for year in [2012, 2016]:  # [2008, 2012, 2016]:
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

                    if presidential_results[1] > presidential_results[0]:
                        result_code = 0
                    else:
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
                republican_vote_share = republican_vote_share + district[1]/(district[1] + district[0])

            republican_vote_share = republican_vote_share / len(all_presidential_results)

            republican_seat_share = 0
            for district in all_state_results:
                if district == 0:
                    republican_seat_share = republican_seat_share + 1

            republican_seat_share = republican_seat_share / len(all_state_results)

            if len(all_state_results) >= min_number_of_districts:
                if (republican_vote_share < 0.5 and republican_seat_share < 0.5) or (
                        1 - republican_vote_share < 0.5 and 1 - republican_seat_share < 0.5):
                    votes.append(republican_vote_share)
                    seats.append(republican_seat_share)

                else:
                    if include_loser_bonus_states:
                        votes.append(republican_vote_share)
                        seats.append(republican_seat_share)
                        loser_bonus_state_count = loser_bonus_state_count + 1
    return votes, seats


# Get overall statewide vote share for each party
def votes_to_overall_vote_share(party_a_votes, party_b_votes):
    party_a_total = np.sum(party_a_votes)
    party_b_total = np.sum(party_b_votes)

    party_a_statewide_vote_share = party_a_total / (party_a_total + party_b_total)
    party_b_statewide_vote_share = party_b_total / (party_a_total + party_b_total)
    return party_a_statewide_vote_share, party_b_statewide_vote_share
