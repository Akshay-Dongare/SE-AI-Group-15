def priority_v2(item, bins_remain_cap):
    priority = []
    for cap in bins_remain_cap:
        if cap >= item:
            priority_score = item / cap  # Favor bins where the item occupies more proportion of the space
            priority.append(priority_score)
        else:
            priority.append(float('-inf'))  # Can't fit the item
    return priority
