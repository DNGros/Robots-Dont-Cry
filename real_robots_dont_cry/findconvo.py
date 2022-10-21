from real_robots_dont_cry.gensurvdatas import get_all_reddit_turn_pairs


def main():
    max_len = 5*15
    all_turns = list(get_all_reddit_turn_pairs())
    print(f"{len(all_turns)=}")
    for turn in all_turns:
        if len(turn.turn_a) > max_len or len(turn.turn_b) > max_len:
            continue
        if turn.turn_a_votes < 50 or turn.turn_b_votes < 100:
            continue
        print(f"Human ({turn.turn_a_votes}): {turn.turn_a}")
        print(f"Resp ({turn.turn_b_votes}): {turn.turn_b}")
        print("-" * 10)


if __name__ == "__main__":
    main()