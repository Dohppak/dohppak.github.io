import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'gatsby';
import styled from '@emotion/styled';

const Small = styled.small({
  color: 'rgba(0,0,0,0.5)',
});

const A = styled(Link)(({ theme }) => ({
  fontWeight: 'normal',
  textDecoration: 'none',
  color: 'rgba(0,0,0,0.5)',
  transition: 'color 250ms linear',
  ':hover': {
    textDecoration: 'underline',
    color: theme.accentColor,
  },
}));

const CommaSeparatedTags = ({ tags }) => (
  <Small>
    {' '}
    {tags.split(', ').map((tag, index, array) => (
      <span key={tag}>
        <A to={`/tag/${tag}/`}>{tag}</A>
        {index < array.length - 1 ? ', ' : ''}
      </span>
    ))}
  </Small>
);

CommaSeparatedTags.propTypes = {
  tags: PropTypes.string,
};

CommaSeparatedTags.defaultProps = {
  tags: '',
};

export default CommaSeparatedTags;
