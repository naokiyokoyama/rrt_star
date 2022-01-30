def RRTStarShortestSelect(rrt_star_pointturn):
    """Allows for dynamic inheritance to choose which class to inherit from upon
    instantiation

    :param rrt_star_pointturn: Must be RRTStarPTSelect inheriting either RRTStarSim or
    RRTStarPNG
    :return: RRTStarShortest class (NOT instance) that inherits from RRTStarPTSelect
    """

    class RRTStarShortest(rrt_star_pointturn):
        cost_key = "best_path_length"
        cost_units = "meters"

        def _cost_from_to(
            self, pt, new_pt, return_heading=False, consider_end_heading=False
        ):
            euclid_dist = self._euclid_2D(pt, new_pt)
            if return_heading:
                _, final_heading = super()._cost_from_to(
                    pt, new_pt, return_heading, consider_end_heading
                )
                return euclid_dist, final_heading

            return euclid_dist

    return RRTStarShortest
